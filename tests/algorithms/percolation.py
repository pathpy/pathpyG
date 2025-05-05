import pathpyG as pp
from pathpyG.algorithms.components import connected_components
from pathpyG.algorithms.temporal import lift_order_temporal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from itertools import product
from collections import Counter
import seaborn as sns

class  TemporalPercolation:
    """
    Implements temporal percolation analysis on temporal networks.

    Attributes:
        temporal_graph (pp.TemporalGraph): The input temporal graph.
        total_events (int): Total number of events in the temporal graph.
    """

    def __init__(self, temporal_graph, total_events):
        """
        Initializes the TemporalPercolation class.

        Args:
            temporal_graph (pp.TemporalGraph): A temporal graph object.
            total_events (int): The total number of events in the dataset.
        """

        self.temporal_graph = temporal_graph
        self.total_events = total_events
        self.N_LT = float(self.temporal_graph.data.time.max().item()) - float(self.temporal_graph.data.time.min().item())
        # for count underlying temporal nodes
        self.src_array = torch.tensor([edge[0] for edge in self.temporal_graph.edges], dtype=torch.long)
        self.dst_array = torch.tensor([edge[1] for edge in self.temporal_graph.edges], dtype=torch.long)

    def create_event_graph(self, delta_t):
        """
        Creates an event graph using temporal edge lifting.

        Args:
            delta_t (float): Time window for lifting edges.

        Returns:
            pp.Graph: The generated event graph.
        """
        event_edge_index = pp.algorithms.temporal.lift_order_temporal(self.temporal_graph, delta=delta_t)
        event_graph = pp.Graph.from_edge_index(event_edge_index)

        return event_graph
    # count the number of underlying temporal nodes, S_G
    def count_underlying_temporal_nodes(self, largest_indices):

        if len(largest_indices) > 0:
            sel_src = self.src_array[largest_indices]
            sel_dst = self.dst_array[largest_indices]
            unique_nodes = np.unique(np.concatenate([sel_src, sel_dst]))
        else:
            unique_nodes = 0
        
        return unique_nodes.size    
    # Lifetime of the event-graph component, S_LT
    def compute_component_lifetime(self, largest_indices):

        if len(largest_indices) > 0:
            times = [float(self.temporal_graph.data.time[i]) for i in largest_indices]
            lifetime  = max(times) - min(times)
        else:
            lifetime  = 0.0  

        return lifetime      

    def percolation_analysis(self, delta_t_values):
        results = {}
        percolation_metrics = {}
        
        for delta_t in delta_t_values:
            print(f"Processing δ={delta_t}...")
            try:
                event_graph = self.create_event_graph(delta_t)
                num_components, labels = pp.algorithms.components.connected_components(event_graph)
                
                # component labels, the value at that index tells us which connected component that node belongs to
                labels_tensor = torch.tensor(labels)
                uniques, counts = torch.unique(labels_tensor, return_counts=True)

                # I also store the component sizes because for XE calculation I need to extract
                # all component sizes except largest one
                # [1,1,1,2,2,3,3,3] -> [3, 2, 3]
                # it turns into a dictionary {3: 2, 2: 1}, component sizes, and their counts
                component_size_counts = dict(Counter(counts.tolist())) 

                if counts.numel() > 0:
                    largest_component_size = counts.max().item()
                    average_component_size = counts.float().mean().item()
                else:
                    largest_component_size = 0
                    average_component_size = 0
                
                # label of largest CC
                if len(uniques) > 0:
                    largest_label = uniques[counts.argmax()].item()
                else:
                    largest_label = -1
                
                # - True (1), the node belongs to the LCC
                # - False (0), it belongs to another component
                largest_indices = (labels_tensor == largest_label).nonzero()

                # the result to a 1D tensor
                # it returns a 2D tensor [[3], [7], [12]]
                # reshapes it into a flat 1D tensor [3, 7, 12]
                largest_indices = largest_indices.view(-1)

                # This list contains the indices of all event-graph nodes that belong to the LCC.
                largest_indices = largest_indices.tolist()  # Example output: [3, 7, 12]

                S_E = len(largest_indices)
                
                # Per-component node sizes and lifetimes
                node_component_sizes = {}
                component_lifetimes = {}

                # same logic as above, but this time I iterate over unique labels to count manually
                for label in uniques.tolist():
                    indices = (labels_tensor == label).nonzero().view(-1).tolist()

                    # Compute S_G (number of unique nodes)
                    node_count = self.count_underlying_temporal_nodes(indices)
                    node_component_sizes[node_count] = node_component_sizes.get(node_count, 0) + 1

                    # Compute S_LT (lifetime of each component)
                    component_times = [self.temporal_graph.data.time[i].item() for i in indices]
                    if component_times:

                        t_min = min(component_times)
                        t_max = max(component_times)
                        # round to get more precise results for lifetime
                        component_lifetime = round(t_max - t_min, 0)  

                        component_lifetimes[component_lifetime] = component_lifetimes.get(component_lifetime, 0) + 1

                S_G = self.count_underlying_temporal_nodes(largest_indices)

                S_LT = self.compute_component_lifetime(largest_indices)
                
                total_components = len(uniques)
                
                
                results[delta_t] = {
                    "largest_component_size": largest_component_size,
                    "total_components": total_components,
                    "average_component_size": average_component_size,
                    "component_sizes": component_size_counts,
                    "node_component_sizes": node_component_sizes,
                    "component_lifetimes": component_lifetimes,
                    "S_E": S_E,
                    "S_G": S_G,
                    "S_LT": S_LT
                }
            
            except Exception as e:
                print(f"Error at δ={delta_t}: {e}")
                results[delta_t] = {
                    "largest_component_size": 0,
                    "total_components": 0,
                    "average_component_size": 0,
                    "component_sizes": 0,
                    "node_component_sizes": 0,
                    "component_lifetimes": 0,
                    "S_E": 0,
                    "S_G": 0,
                    "S_LT": 0
                }
        
        return results, percolation_metrics

    def find_critical_threshold(self, analysis_results, delta_t_values):
            """
            Identify the critical δt where the largest connected component (LCC) rapidly grows.
            """
            
            num_event_graph_nodes = np.array([result["S_E"] for result in analysis_results.values()])

            if len(delta_t_values) < 2:
                raise ValueError("Not enough data points to determine the critical threshold.")

            susceptibility = np.diff(num_event_graph_nodes) / np.diff(delta_t_values)

            # I added because it made an error in the length of the susceptibility
            # so, the first and last values are removed in the derivative calculation
            midpoints = (delta_t_values[:-1] + delta_t_values[1:]) / 2

            critical_index = np.argmax(susceptibility)
            critical_dt = midpoints[critical_index]

            return critical_dt


    def compare_component_measures(self, results):
        """Compare S_E, S_G, and S_LT by plotting them across delta_t values with S_max notation."""
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df[['S_E', 'S_G', 'S_LT']]
        df = (df - df.min()) / (df.max() - df.min())  # Normalize for fair comparison

        seconds_per_day = 24 
        df.index = df.index.astype(float) / seconds_per_day

        # Set academic-style theme
        sns.set_theme(style="whitegrid")

        # Create figure
        plt.figure(figsize=(10, 6))

        # Line plots with distinct markers
        plt.plot(df.index, df['S_E'], marker='o', linestyle='-', markersize=6, label=r'$S_{\max, E}$ (Largest Event Graph Component)')
        plt.plot(df.index, df['S_G'], marker='s', linestyle='--', markersize=6, label=r'$S_{\max, G}$ (Largest Underlying Graph Component)')
        plt.plot(df.index, df['S_LT'], marker='^', linestyle='-.', markersize=6, label=r'$S_{\max, LT}$ (Largest Component Lifetime)')

        # Labels and formatting
        plt.xlabel(r"$\delta_t$ (days)", fontsize=14)
        plt.ylabel("Normalized Largest Component Size " + r"$S_{\max}$", fontsize=14)
        plt.title("Comparison of Largest Component Sizes Across Measures", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.xlim(0, 25)
        # Show the plot
        plt.show()

        # Compute correlation matrix
        correlation_matrix = df.corr()
        print("\nCorrelation between measures:")
        print(correlation_matrix)

        plt.figure(figsize=(6, 5))
        sns.set_theme(style="white")

        # Plot heatmap with updated labels
        ax = sns.heatmap(
            correlation_matrix,
            annot=True, 
            cmap="Blues",
            cbar=True,
            square=True,
            linewidths=0.5,
            fmt=".2f"
        )

        ax.set_xticklabels([r"$S_{\max, E}$", r"$S_{\max, G}$", r"$S_{\max, LT}$"], fontsize=12, fontstyle="italic")
        ax.set_yticklabels([r"$S_{\max, E}$", r"$S_{\max, G}$", r"$S_{\max, LT}$"], fontsize=12, fontstyle="italic")

        ax.set_xlabel(r"$d_2$", fontsize=14, fontstyle="italic")
        ax.set_ylabel(r"$d_1$", fontsize=14, fontstyle="italic")

        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        # Save figure
        plt.savefig("correlation_matrix.png", dpi=300, bbox_inches='tight')

        plt.show()


    def plot_largest_component(self, results, delta_t_values):
            
            df = pd.DataFrame.from_dict(results, orient='index')
            df = df[['S_E']]
            num_event_graph_nodes = df["S_E"]

            delta_t_values = delta_t_values 
            critical_delta_t = self.find_critical_threshold(results, delta_t_values)
            
            plt.figure(figsize=(10, 6))
            plt.plot(delta_t_values, num_event_graph_nodes, label="Number of Events", marker="o")
            plt.axvline(x=critical_delta_t, color="r", linestyle="--", label=f"Critical δt={critical_delta_t:.2f} hours")
            
            plt.xlabel("δ_t (hours)")
            plt.ylabel("Event Graph Nodes")
            plt.title("Phase Transition Analysis")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.ticklabel_format(style='plain')  
            plt.show()

    def compute_order_parameters(self, results, temporal_graph):
        """
        Compute order parameters (ρE, ρG), susceptibilities (χE, χG), 
        and probability distributions P(S_E) and P(S_LT) using the component size dictionaries.
        """

        delta_t_values = np.array([float(dt) for dt in results.keys()])
        
        maxS_E_values = np.array([results[str(dt)]["S_E"] for dt in delta_t_values])
        maxS_G_values = np.array([results[str(dt)]["S_G"] for dt in delta_t_values])
        maxS_LT_values = np.array([results[str(dt)]["S_LT"] for dt in delta_t_values]) 
        N_values = np.array([results[str(dt)]["total_components"] for dt in delta_t_values])
        N_E = self.total_events  # Total number of nodes in the temporal network
        all_nodes = np.unique(np.concatenate([self.src_array, self.dst_array]))
        N_G = len(all_nodes)  # Total number of unique nodes

        # Order Parameters:
        # - ρ_E: Fraction of event-graph nodes in the largest component
        # - ρ_G: Fraction of temporal-network nodes in the largest component
        rho_E = np.where(N_values > 0, maxS_E_values / N_E, 0)  # Prevent divide-by-zero
        rho_G = np.where(N_values > 0, maxS_G_values / N_G, 0)
        rho_LT = np.where(N_values > 0, maxS_LT_values / self.N_LT, 0)
        print("Total Values:")
        print(f"NE: {N_E}, NG: {N_G}, NLT: {self.N_LT}")
        # Initialize susceptibility arrays
        chi_E = np.zeros_like(delta_t_values, dtype=np.float64)
        chi_G = np.zeros_like(delta_t_values, dtype=np.float64)
        chi_LT = np.zeros_like(delta_t_values, dtype=np.float64)

        # Initialize distributions for P(S_E) and P(S_LT)
        P_S_E_distributions = {}
        P_S_LT_distributions = {}
        P_S_G_distributions = {}

        for i, dt in enumerate(delta_t_values):

            # χE Computation (Event-Based Susceptibility) 
            component_size_counts = results[str(dt)].get("component_sizes", {})
            if component_size_counts == 0:
                component_size_counts = {}  # Replace invalid value

            # χG Computation (Node-Based Susceptibility) 
            node_component_sizes = results[str(dt)].get("node_component_sizes", {})

            # S_LT Computation (Lifetime-Based Susceptibility) 
            component_lifetime_counts = results[str(dt)].get("component_lifetimes", {})

            if not component_size_counts and not node_component_sizes and not component_lifetime_counts:
                continue  # Skip if no valid components

            # χE Calculation
            if component_size_counts:
                component_size_counts = {int(k): v for k, v in component_size_counts.items()}

                component_sizes = np.array(list(component_size_counts.keys()))
                counts = np.array(list(component_size_counts.values()))

                largest_component_size = component_sizes.max(initial=0)
                if largest_component_size > 0:
                    mask = component_sizes < largest_component_size
                    s_squared_counts = (component_sizes[mask] ** 2) * counts[mask]
                    
                    # Compute susceptibility for χ_E (event-based)
                    chi_E[i] = s_squared_counts.sum() / N_E

                # Compute probability distribution P(S_E)
                total_S_E_components = sum(component_size_counts.values())
                P_S_E = {size: count / total_S_E_components for size, count in component_size_counts.items()}
                P_S_E_distributions[dt] = P_S_E

            # χG Calculation
            if node_component_sizes:
                node_component_sizes = {int(k): v for k, v in node_component_sizes.items()}

                node_sizes = np.array(list(node_component_sizes.keys()))
                node_counts = np.array(list(node_component_sizes.values()))

                largest_node_component = node_sizes.max(initial=0)
                if largest_node_component > 0:
                    node_mask = node_sizes < largest_node_component
                    s_squared_counts_nodes = (node_sizes[node_mask] ** 2) * node_counts[node_mask]
                    
                    # Compute susceptibility for χ_G (node-based)
                    chi_G[i] = s_squared_counts_nodes.sum() / N_G
                # Compute probability distribution P(S_G)
                total_S_G_components = sum(node_component_sizes.values())
                P_S_G = {size: count / total_S_G_components for size, count in node_component_sizes.items()}
                P_S_G_distributions[dt] = P_S_G
            
            # χLT Calculation
            if component_lifetime_counts:
                component_lifetime_counts = {(float(k)): v for k, v in component_lifetime_counts.items()}

                # Extract component sizes and their counts
                component_lifetimes = np.array(list(component_lifetime_counts.keys()))
                counts = np.array(list(component_lifetime_counts.values()))

                # Identify the largest component
                largest_lifetime = component_lifetimes.max(initial=0)
                if largest_lifetime > 0:
                    mask = component_lifetimes < largest_lifetime
                    s_squared_counts_lt = (component_lifetimes[mask] ** 2) * counts[mask]

                    # Compute susceptibility for χ_LT (lifetime-based)
                    chi_LT[i] = s_squared_counts_lt.sum() / self.N_LT  # N_LT = T

                    # Compute probability distribution P(S_LT)
                    total_S_LT_components = sum(component_lifetime_counts.values())
                    P_S_LT = {size: count / total_S_LT_components for size, count in component_lifetime_counts.items()}
                    P_S_LT_distributions[dt] = P_S_LT
    
        # If there is NaN values with zero, replace
        chi_E = np.nan_to_num(chi_E)
        chi_G = np.nan_to_num(chi_G)
        chi_LT = np.nan_to_num(chi_LT)


        return delta_t_values, rho_E, rho_G, rho_LT, chi_E, chi_G, chi_LT, P_S_E_distributions, P_S_LT_distributions, P_S_G_distributions

    def plot_percolation_metrics(self, delta_t_values, rho_metrics, chi_metrics, labels, colors, time_range=0):
        """
        Function to plot percolation metrics with dual y-axes.

        Parameters:
        delta_t_values: numpy array of time intervals in hours
        rho_metrics: list of rho metric arrays
        chi_metrics: list of chi metric arrays
        labels: list of labels for rho and chi metrics
        colors: tuple of colors for left (rho) and right (chi) axes
        """
        delta_t_hours = delta_t_values 
        fig, axes = plt.subplots(nrows=len(rho_metrics), figsize=(8, 10), sharex=True)

        for i in range(len(rho_metrics)):
            ax = axes[i]
            ax_twin = ax.twinx()

            sns.lineplot(x=delta_t_hours, y=rho_metrics[i], ax=ax, marker='o', markersize=6, label=labels[i][0], color=colors[0], linewidth=2)
            sns.lineplot(x=delta_t_hours, y=chi_metrics[i], ax=ax_twin, marker='s', markersize=6, label=labels[i][1], color=colors[1], linewidth=2)

            ax.set_ylabel(labels[i][0], color=colors[0], fontsize=16)  
            ax_twin.set_ylabel(labels[i][1], color=colors[1], fontsize=16)

            ax.yaxis.label.set_color(colors[0])
            ax_twin.yaxis.label.set_color(colors[1])
            ax_twin.tick_params(axis='y', colors=colors[1], labelsize=14)  
            ax.tick_params(axis='y', colors=colors[0], labelsize=14)  
            ax.tick_params(axis='x', labelsize=14)  

            if i == 2:
                ax.legend(loc="upper right", bbox_to_anchor=(1, 0.8), fontsize=14)  
                ax_twin.legend(loc="upper right", bbox_to_anchor=(1, 0.6), fontsize=14)  
            else:
                ax.legend(loc="upper right", bbox_to_anchor=(1, 0.7), fontsize=14)  
                ax_twin.legend(loc="upper right", bbox_to_anchor=(1, 0.5), fontsize=14)  

        sns.despine()
        
        axes[-1].set_xlabel(r'$\delta t$ (in hours)', fontsize=20)  
        axes[0].set_xlim(0, time_range)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    # I added this function for finite size scaling, but I did not use
    def finite_size_scaling(self, analysis_results, delta_t_values, sizes, critical_delta_t, beta=0.5, gamma=2/3):
        """
        X-axis: N^(1/2) * (δt - δt_c)
        Y-axis: N^(β/2) * ρ_E(δt - δt_c)
        """
        scaled_results = {}
        total_n = self.temporal_graph.n
        for size in sizes:
            largest_components = np.array([analysis_results[dt]["largest_component_size"] for dt in delta_t_values])
    
            # (ρ_E = S/N) # N* is the maximum possible value that S* can get as a single component
            rho_E = largest_components / total_n

            # N^(1/2) * (δt - δt_c)
            scaled_delta_t = (delta_t_values - critical_delta_t) * (size ** (1 / 2))

            # ρ_E(δt - δt_c) * N^(β/2) 
            scaled_largest =  rho_E * (size ** (beta / 2))

            scaled_results[size] = {
                "scaled_delta_t": scaled_delta_t,
                "scaled_largest_component": scaled_largest,
            }

        return scaled_results
        