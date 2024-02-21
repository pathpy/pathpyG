console.log("Static Network Template");
/* Resources
   https://bl.ocks.org/mapio/53fed7d84cd1812d6a6639ed7aa83868
   https://codepen.io/smlo/pen/JdMOej
*/

// variables from the config file
const selector = config.selector;
const width = config.width || 800;
const height = config.height || 600;
const charge_distance = config.charge_distance || 400;
const charge_force = config.charge_force || -3000;
const curved = config.curved || false;
const directed = config.directed || false;
// const weight = false;

/* Create a svg element to display the network */
var svg = d3.select(selector)
    .append('svg')
    .attr('width', width)
    .attr('height', height)

// add container to store the elements
var container = svg.append("g");

/*Add zoom function to the container */
svg.call(
    d3.zoom()
        .scaleExtent([.1, 4])
        .on("zoom", function() { container.attr("transform", d3.event.transform); })
);


/*Load nodes and links from the data */
var nodes = data.nodes
var links = data.edges

/*Create arrow head with same color as the edge */
function marker (color) {
       var reference;
       svg.append("svg:defs").selectAll("marker")
          .data([reference])
          .enter().append("svg:marker")
          .attr("id", "arrow"+color)
          .attr("viewBox", "0 -5 10 10")
          .attr("refX", 10)
          .attr("refY", -0)
          .attr("markerWidth", 6)
          .attr("markerHeight", 6)
          .attr("orient", "auto")
          .append("svg:path")
          .attr('class','.arrow')
          .attr("d", "M0,-5L10,0L0,5")
          .style('opacity',1)
          .style("fill", color);
       return "url(#" + "arrow"+color + ")";
     };

/*Link creation template */
var link = container.append("g").attr("class", "links")
    .selectAll(".link")
    .data(links)
    .enter()
    .append("path")
    .attr("class", "link")
    .style("stroke", function(d) { return d.color; })
    .style("stroke-opacity", function(d) { return d.opacity; })
    .style("stroke-width", function(d){  return d.size })
    .style("fill","none")
    .attr("marker-end", function (d) {if(directed){return marker(d.color)}else{return null}; });

    //.attr("marker-end", function (d) { return marker(d.color); });
    //.attr("marker-end", "url(#arrow)");

/*Node creation template */
var node = container.append("g").attr("class", "nodes")
    .selectAll("circle.node")
    .data(nodes)
    .enter().append("circle")
    .attr("class", "node")
    .attr("x", function(d) { return d.x; })
    .attr("y", function(d) { return d.y; })
    .style("r", function(d){  return d.size+"px"; })
    .style("fill", function(d) { return d.color; })
    .style("opacity", function(d) { return d.opacity; });

/*Label creation template */
var text = container.append("g").attr("class","labels")
    .selectAll("g")
    .data(nodes)
    .enter().append("g")

text.append("text")
    .attr("class", "label-text")
    .attr("x", function(d) {
        var r = (d.size === undefined) ? 15 : d.size;
        return 5 + r; })
    .attr("dy", ".31em")
    .text(function(d) { return d.label; });

/*Scale weight for d3js */
var weightScale = d3.scaleLinear()
    .domain(d3.extent(links, function (d) { return d.weight }))
    .range([.1, 1]);

/*Simulation of the forces*/
var simulation = d3.forceSimulation(nodes)
    .force("links", d3.forceLink(links)
           .id(function(d) {return d.uid; })
           .distance(50)
           .strength(function(d){return weightScale(d.weight);})
          )
    .force("charge", d3.forceManyBody()
           .strength(charge_force)
           .distanceMax(charge_distance)
          )
    .force("center", d3.forceCenter(width / 2, height / 2))
    .on("tick", ticked);

/*Update of the node and edge objects*/
function ticked() {
    node.call(updateNode);
    link.call(updateLink);
    text.call(updateText);
};

/*Update link positions */
function updateLink(link) {
    // link
    //     .attr("x1", function(d) { return d.source.x; })
    //     .attr("y1", function(d) { return d.source.y; })
    //     .attr("x2", function(d) { return d.target.x; })
    //     .attr("y2", function(d) { return d.target.y; });


    link.attr("d", function(d) {
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy);
        if(!curved)dr=0;
        return "M" +
            d.source.x + "," +
            d.source.y + "A" +
            dr + "," + dr + " 0 0,1 " +
            d.target.x + "," +
            d.target.y;
    });

    // recalculate and back off the distance
    link.attr("d", function (d, i) {
        var pl = this.getTotalLength();
        var r = (d.target.size === undefined) ? 15 : d.target.size;
        var m = this.getPointAtLength(pl - r);
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy);
        if(!curved)dr=0;
        var result = "M" + d.source.x + "," + d.source.y + "A" + dr + "," + dr + " 0 0,1 " + m.x + "," + m.y;
        return result;
    });
};


/*Update node positions */
function updateNode(node) {
    node.attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
    });
    // node
    //     .attr("cx", function(d) { return d.x; })
    //     .attr("cy", function(d) { return d.y; });
};

/*Update text positions */
function updateText(text) {
    text.attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
    });
};

/*Add drag functionality to the node objects*/
node.call(
    d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
);

function dragstarted(d) {
    d3.event.sourceEvent.stopPropagation();
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
};

function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
};

function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
};
