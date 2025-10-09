// Useful pages
// ------------
// https://github.com/takanori-fujiwara/d3-gallery-javascript/tree/main/animation/temporal-force-directed-graph
// https://observablehq.com/@d3/temporal-force-directed-graph
// https://observablehq.com/@mbostock/scrubber
// https://plnkr.co/edit/0VxwY1Mc5UYusvhYic8u?preview
// https://stackoverflow.com/questions/42545531/making-d3-like-force-directed-graph-with-pngs


// Create function for Network Drawing
const Network = (config) => {
    console.log("Initialize Network Function");

    // Get variables from config
    const selector = config.selector;       // DOM uuid
    const width = config.width || 800;      // window width 
    const height = config.height || 600;    // window height
    const delta = config.delta || 300;      // time between frames
    const padding = (config.node && config.node.image_padding) || 5;    // distance between node and image
    const margin = config.margin || 0.1;  // margin around the plot area for fixed layout
    const xlim = [-1*margin, 1+(1*margin)]; // limits of the x-coordinates
    const ylim = [-1*margin, 1+(1*margin)]; // limits of the y-coordinates
    const arrowheadMultiplier = 4; // Multiplier for arrowhead size based on edge stroke width
    const nodeStrokeWidth = 2.5; // Stroke width around nodes

    // Initialize svg canvas
    const svg = d3.select(selector)
          .append('svg')
          .attr('width', width)
          .attr('height', height)
          .attr('viewBox', [0, 0, width, height]);
    
    // add container to store network
    let container = svg.append("g");

    // initialize link
    let link = container.append("g")
        .attr("class", "edges")
        .selectAll(".link");

    // initialize node
    let node = container.append("g")
        .attr("class", "nodes")
        .selectAll("circle.node");

    // initialize label
    let label = container.append("g")
        .attr("class", "labels")
        .selectAll(".label");

    // initialize image
    let image = container.append("g")
        .attr("class", "images")
        .selectAll(".image");

    // Helper function to calculate and store link endpoints on the data object
    const calculateAndStoreLinkPath = (d) => {
        const source_x = d.source.x;
        const source_y = d.source.y;
        const target_x = d.target.x;
        const target_y = d.target.y;

        const sourceRadius = d.source.size || (config.node && config.node.size) || 15;
        const targetRadius = d.target.size || (config.node && config.node.size) || 15;
        
        let effectiveSourceRadius = sourceRadius + nodeStrokeWidth / 2;
        let effectiveTargetRadius = targetRadius + nodeStrokeWidth / 2;

        let finalPath;

        if (config.directed) {
            // --- For CURVED Directed Links (Complex Path) ---
            const dx = target_x - source_x;
            const dy = target_y - source_y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Create a virtual path from center to center
            const virtualPathData = `M${source_x},${source_y} A${distance},${distance} 0 0,1 ${target_x},${target_y}`;

            // Use a temporary path element to measure
            const tempPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
            tempPath.setAttribute("d", virtualPathData);
            
            const pathLength = tempPath.getTotalLength();
            
            // Adjust effective radii to account for arrowhead size
            const edgeStrokeWidth = d.size || (config.edge && config.edge.size) || 2;
            const arrowheadLength = edgeStrokeWidth * arrowheadMultiplier;
            effectiveTargetRadius += arrowheadLength;
            
            // Find the precise intersection points by moving along the path
            const startPoint = tempPath.getPointAtLength(effectiveSourceRadius);
            const endPoint = tempPath.getPointAtLength(pathLength - effectiveTargetRadius);

            // Rebuild the arc with the new, correct endpoints
            const newDx = endPoint.x - startPoint.x;
            const newDy = endPoint.y - startPoint.y;
            const newDistance = Math.sqrt(newDx * newDx + newDy * newDy);
            
            finalPath = `M${startPoint.x},${startPoint.y} A${newDistance},${newDistance} 0 0,1 ${endPoint.x},${endPoint.y}`;

        } else {
            // --- For STRAIGHT Undirected Links (Simple Path) ---
            const dx = target_x - source_x;
            const dy = target_y - source_y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance === 0) {
                 d._path = `M${source_x},${source_y} L${target_x},${target_y}`;
                 return;
            }
            
            const x1 = source_x + (dx / distance) * effectiveSourceRadius;
            const y1 = source_y + (dy / distance) * effectiveSourceRadius;
            const x2 = target_x - (dx / distance) * effectiveTargetRadius;
            const y2 = target_y - (dy / distance) * effectiveTargetRadius;
            finalPath = `M${x1},${y1} L${x2},${y2}`;
        }
        
        d._path = finalPath;
    };

    const ticked = () => {
        // First, iterate through the link data to calculate all endpoints (and curves if directed)
        // so that the link only touches the edge of the node when opacity is < 1
        link.data().forEach(calculateAndStoreLinkPath);

        node.call(updateNodePosition);
        link.call(updateLinkPosition);
        if (config.show_labels) {
            label.call(updateLabelPosition);
        }
        image.call(updateImagePosition);
    }

    // update node position
    const updateNodePosition = (node) => {
        node.attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });
    };

    // update label position
    const updateLabelPosition = (label) => {
        label.attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });
    };

    // update image position
    const updateImagePosition = (image) => {
        image.attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        });
    };

    // update link position
    const updateLinkPosition = (link) => {
        link.attr('d', d => d._path);
    };

    const simulation = d3.forceSimulation()
          .velocityDecay(0.2)
          .alphaMin(0.1)
          .force('link', d3.forceLink().id(d => d.uid))
          .on('tick', ticked);

    let currentlyDragged = null; // Remember currently dragged node during update

    // Add drag functionality to the node objects
    const drag = d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended);

    function dragstarted(event, d) {
        currentlyDragged = d;
        event.sourceEvent.stopPropagation();
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    };

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    };

    function dragended(event, d) {
        currentlyDragged = null;
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    };

    /**
    * Creates a custom D3 force that repels nodes from a rectangular boundary.
    * @param {number} x0 - The left boundary.
    * @param {number} y0 - The top boundary.
    * @param {number} x1 - The right boundary.
    * @param {number} y1 - The bottom boundary.
    * @param {number} strength - The strength of the repulsion.
    */
    function forceBoundary(x0, y0, x1, y1, strength = 0.5) {
        let nodes;

        function force(alpha) {
            for (let i = 0, n = nodes.length; i < n; ++i) {
                const node = nodes[i];
                const r = node.size || (config.node && config.node.size) || 15;

                // Push node away from the left boundary
                if (node.x - r < x0) {
                    node.vx += (x0 - (node.x - r)) * strength * alpha;
                }
                // Push node away from the right boundary
                if (node.x + r > x1) {
                    node.vx += (x1 - (node.x + r)) * strength * alpha;
                }
                // Push node away from the top boundary
                if (node.y - r < y0) {
                    node.vy += (y0 - (node.y - r)) * strength * alpha;
                }
                // Push node away from the bottom boundary
                if (node.y + r > y1) {
                    node.vy += (y1 - (node.y + r)) * strength * alpha;
                }
            }
        }

        force.initialize = function(_) {
            nodes = _;
        };

        return force;
    }

    return Object.assign(svg.node(), {
        update({nodes, links}) {

            // --- DATA PREPARATION ---
            // Preserve node positions across updates
            const oldNodesMap = new Map(node.data().map(d => [d.uid, d]));
            nodes = nodes.map(newNode => {
                const oldNode = oldNodesMap.get(newNode.uid);
                
                // Check if this is the node currently being dragged
                if (currentlyDragged && currentlyDragged.uid === newNode.uid) {
                    // If so, update the original object in place...
                    Object.assign(currentlyDragged, newNode);
                    // ...and return the original object to preserve its identity.
                    return currentlyDragged;
                }
                
                // For all other nodes, create a new merged object as before.
                return { ...oldNode, ...newNode };
            });
            links = links.map(d => ({...d}));

            // --- NODES (CIRCLES) ---
            // 1. Data Join
            node = container.select('.nodes').selectAll("circle.node")
                .data(nodes, d => d.uid);

            // 2. Exit Selection: Fade out and remove old nodes
            node.exit()
                .transition()
                .duration(delta / 2)
                .style("opacity", 0)
                .remove();

            // 3. Enter & Merge: Create new circles and merge with updating ones
            node = node.enter().append('circle')
                .attr("class", "node")
                .call(drag)
                .merge(node);

            // 4. Update Selection: Apply transitions to all nodes (new and existing)
            node.transition()
                .duration(delta)
                .style("r", d => (d.size || (config.node && config.node.size)) + "px") // Use fallback for size
                .style("fill", d => (d.color || (config.node && config.node.color))) // Use fallback for color
                .style("opacity", d => (d.opacity || (config.node && config.node.opacity))) // Use fallback for opacity
                .style("stroke-width", nodeStrokeWidth + "px")
                .style("stroke", "#000000");

            // --- IMAGES ---
            // 1. Data Join
            image = container.select('.images').selectAll('.image')
                .data(nodes.filter(d => d.image), d => d.uid);

            // 2. Exit Selection: Fade out and remove old images
            image.exit()
                .transition()
                .duration(delta / 2)
                .style("opacity", 0)
                .remove();

            // 3. Enter & Merge: Create new images and merge with updating ones
            image = image.enter().append('image')
                .attr("class", "image")
                .attr("xlink:href", d => d.image)
                .style("width", "0px")
                .style("height", "0px")
                .call(drag)
              .merge(image);

            // 4. Update Selection: Apply transitions to all images
            image.transition()
                .duration(delta)
                .attr("x", d => -(d.size || (config.node && config.node.size)) + padding)
                .attr("y", d => -(d.size || (config.node && config.node.size)) + padding)
                .style("width", d => 2 * (d.size || (config.node && config.node.size)) - 2 * padding + "px")
                .style("height", d => 2 * (d.size || (config.node && config.node.size)) - 2 * padding + "px");


            // --- LABELS ---
            if (config.show_labels) {
                // 1. Data Join
                label = container.select('.labels').selectAll('.label-text')
                    .data(nodes, d => d.uid);

                // 2. Exit Selection: Fade out and remove old labels
                label.exit()
                    .transition()
                    .duration(delta / 2)
                    .style("opacity", 0)
                    .remove();

                // 3. Enter & Merge: Create new text elements and merge with updating ones
                label = label.enter().append('text')
                    .attr("class", "label-text")
                    .attr("dy", ".32em")
                    .style("opacity", 0) // Start transparent to fade in
                .merge(label);

                // 4. Update Selection: Apply transitions to all labels
                label.transition()
                    .duration(delta)
                    .attr("x", d => (d.size || 15) + 5) // Use fallback for size
                    .text(d => d.uid)
                    .style("opacity", 1); // Fade in new/updating labels
            }

            // --- DYNAMIC ARROWHEAD MARKERS ---
            if (config.directed) {
                // Ensure a <defs> element exists
                const defs = container.selectAll('defs').data([1]).join('defs');

                // 1. Get a list of all unique edge colors currently in the data
                const uniqueColors = Array.from(
                    new Set(links.map(d => d.color || (config.edge && config.edge.color)))
                );

                // 2. Perform a data join to create one marker per unique color
                const markers = defs.selectAll('marker')
                    .data(uniqueColors, color => color); // Key the data by the color string itself

                // 3. Remove any markers for colors that are no longer in the data
                markers.exit().remove();

                // 4. For any new colors, create a new marker
                markers.enter().append('marker')
                    .attr('id', color => `arrowhead-${color.replace('#', '')}`) // e.g., "arrowhead-ff0000"
                    .attr('viewBox', '0 -5 10 10')
                    .attr('refX', 0)
                    .attr('refY', 0)
                    .attr('markerUnits', 'strokeWidth')
                    .attr('markerWidth', arrowheadMultiplier)
                    .attr('markerHeight', arrowheadMultiplier)
                    .attr('orient', 'auto')
                    .append('path')
                        .attr('d', 'M0,-5L10,0L0,5')
                        .style('fill', color => color); // Set the fill using the color data
            }


            // --- LINKS (EDGES) ---
            // 1. Data Join
            link = container.select(".edges").selectAll(".link")
                .data(links, d => d.uid);

            // 2. Exit Selection
            link.exit().remove();

            // 3. Enter & Merge
            link = link.enter().append("path")
                .attr("class", "link")
                .style("fill", "none")
                .merge(link);

            // 4. Update Selection
            link.transition()
                .duration(delta)
                .style("stroke", d => (d.color || (config.edge && config.edge.color)))
                .style("color", d => (d.color || (config.edge && config.edge.color))) // For arrowhead color
                .style("stroke-width", d => (d.size || (config.edge && config.edge.size)) + 'px')
                .style("opacity", d => (d.opacity || (config.edge && config.edge.opacity)));

        // Conditionally add the correct arrowhead marker
        if (config.directed) {
            link.attr('marker-end', d => {
                // Find the color for this specific link
                const color = d.color || (config.edge && config.edge.color);
                // Create the safe ID that matches the marker definition
                const safeColorId = color.replace('#', '');
                // Return the URL pointing to the specific marker
                return `url(#arrowhead-${safeColorId})`;
            });
        } else {
            link.attr('marker-end', null);
        }

            simulation.nodes(nodes);
            simulation.force("link").links(links);

            // Based on the config.simulation parameter, choose the layout type.
            if (config.simulation) {
                // TRUE: Use a dynamic spring layout (force-directed)
                simulation
                    .force('charge', d3.forceManyBody().strength(-50)) // Nodes repel each other
                    .force('center', d3.forceCenter(width/2, height/2)) // Center the graph
                    .force('x', null) // Remove the static x-force
                    .force('y', null) // Remove the static y-force
                    .force('boundary', forceBoundary(0, 0, width, height));
                
                // Adjust link force
                simulation.force("link").strength(0.1).distance(70);

            } else {
                // FALSE: Use the original fixed layout based on xpos and ypos
                simulation.force('charge', d3.forceManyBody().strength(-20)); // Weak charge to prevent some overlap
                simulation.force('center', null); // No need for centering force
                simulation.force('boundary', null); 

                // Use x/y forces to position nodes based on data
                const xScale = d3.scaleLinear().domain(xlim).range([0, width]);
                const yScale = d3.scaleLinear().domain(ylim).range([0, height]);
                simulation.force('x', d3.forceX().strength(0.1).x(d => xScale(d.xpos)));
                simulation.force('y', d3.forceY().strength(0.1).y(d => yScale(d.ypos)));
                
                // Weaken link force so it doesn't fight the x/y positioning
                simulation.force("link").strength(0);
            }

            // Restart simulation and render immediately
            simulation.alpha(1).restart().tick();
            ticked();
        }
    });
}; // End Network
