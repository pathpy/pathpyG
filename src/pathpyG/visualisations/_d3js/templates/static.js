// Initialize Network
const network = Network(config);


const drawStaticNetwork = () => {
    // Get all nodes and links from the data object
    const nodes = data.nodes;
    const links = data.edges;
    // Call the network's update function once with the complete dataset
    network.update({nodes, links});
};

// Draw the network.
drawStaticNetwork();
