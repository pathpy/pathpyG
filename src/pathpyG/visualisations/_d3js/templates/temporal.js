console.log("Temporal Network Template");
/* Resources
   https://bl.ocks.org/mapio/53fed7d84cd1812d6a6639ed7aa83868
   https://codepen.io/smlo/pen/JdMOej
   https://observablehq.com/@d3/temporal-force-directed-graph
*/

// console.log(data);

// variables from the config file
const selector = config.selector;
const width = config.width || 800;
const height = config.height || 600;
const delta = config.delta || 300;

// variables for the temporal components
const startTime = config.start;
const endTime = config.end;
const targetValue = config.intervals || 300;
const duration = config.delta || 300;  

// variables for the edge components
const curved = config.curved || false;
const directed = config.directed || false;

/* Create a svg element to display the network */
let svg = d3.select(selector)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

/*Container to store d3js objects */
let container = svg.append("g");

/*Link creation template */
let edges = container.append("g").attr("class", "edges")
    .selectAll(".link");

/*Node creation template */
let nodes = container.append("g").attr("class", "nodes")
    .selectAll("circle.node");

/*Label creation template */
let labels = container.append("g").attr("class", "labels")
    .selectAll(".label");

/*Time counter */
let text = svg.append("text")
    .text("T="+startTime)
    .attr("x", 20)
    .attr("y", 20);

let bttn = svg.append("text")
    .attr("x",70)
    .attr("y", 20)
    .text("Play");

/*Assign data to variable*/
let network = data


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

/*Render function to show dynamic networks*/
function render(){

    // get network data
    let nodeData = network.nodes;
    let edgeData = network.edges;
    // let labelData = network.nodes;

    // render network objects
    renderNodes(nodeData);
    renderEdges(edgeData);
    renderLabels(nodeData);

    // run simulation
    simulation.nodes(nodeData);
    simulation.force("links").links(edgeData);
    simulation.alpha(1).restart();
}

/*Render node objects*/
function renderNodes(data){
    // console.log("render Nodes")
    
    nodes = container.select('.nodes').selectAll('circle.node').data(data,d=>d.uid);

    let new_nodes = nodes.enter().append("circle")
        .attr("class", "node")
        .style("r", function(d){  return d.size+"px"; })
        .style("fill", d => d.color)
        .style("opacity", d => d.opacity)
        .call(drag);
    
    nodes.exit()
        .transition() // transition to shrink node
        .duration(delta)
        .style("r", "0px")
        .remove();
    
    nodes = nodes.merge(new_nodes);

    nodes.transition() // transition to change size and color
        .duration(delta)
        .style("r", function(d){  return d.size+"px"; })
        .style("fill", d => d.color)
        .style("opacity", d => d.opacity);
};

/*Render label objects*/
function renderLabels(data){
    // console.log("render Nodes")
    
    labels = container.select('.nodes').selectAll('.label-text').data(data, d=> d.uid);

    let new_labels = labels.enter().append("text")
        .attr("class", "label-text")
        .attr("x", function(d) {
            var r = (d.size === undefined) ? 15 : d.size;
            return 5 + r; })
        .attr("dy", ".32em")
        .text(d=>d.label);
    
    labels.exit().remove();

    labels = labels.merge(new_labels);
};

/*Render edge objects*/
function renderEdges(data){
    // console.log("render Edges")
    edges = container.select(".edges").selectAll(".link").data(data, d=> d.uid);

    let new_edges =  edges.enter().append("path")
        .attr("class", "link")
        .style("stroke", d => d.color)
        .style("stroke-opacity", d => d.opacity)
        .style("stroke-width", d => d.size)
        .style("fill","none")
        .attr("marker-end", function (d) {if(directed){return marker(d.color)}else{return null}; });

    edges.exit().remove();

    edges = edges.merge(new_edges);

    edges.transition() // transition to change size and color
        .duration(delta)
        .style("stroke", d => d.color)
        .style("stroke-opacity", d => d.opacity)
        .style("stroke-width", d => d.size);
};

/*Add zoom function to the container */
svg.call(
    d3.zoom()
        .scaleExtent([.1, 4])
        .on("zoom", function() { container.attr("transform", d3.event.transform); })
).on("dblclick.zoom", null);


/*Simulation of the forces*/
const simulation = d3.forceSimulation()
      .force("charge", d3.forceManyBody().strength(-3000))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("x", d3.forceX(width / 2).strength(1))
      .force("y", d3.forceY(height / 2).strength(1))
      .force("links", d3.forceLink()
             .id( d => d.uid)
             .distance(50).strength(1))
      .on("tick", ticked);


/*Update of the node and edge objects*/
function ticked() {
    nodes.call(updateNodePositions);
    edges.call(updateEdgePositions);
    labels.call(updateLabelPositions);
};

/*Update link positions */
function updateEdgePositions(edges) {
    // edges
    //     .attr("x1", d => d.source.x)
    //     .attr("y1", d => d.source.y)
    //     .attr("x2", d => d.target.x)
    //     .attr("y2", d => d.target.y);

    edges.attr("d", function(d) {
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
    edges.attr("d", function (d, i) {
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
function updateNodePositions(nodes) {
    nodes.attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
    });
    // nodes
    //     .attr("cx", d => d.x)
    //     .attr("cy", d => d.y);
};


/*Update node positions */
function updateLabelPositions(labels) {
    labels.attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
    });
};

/*Add drag functionality to the node objects*/
const drag = d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);

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


/*Temporal components*/
let currentValue = 0;
let time = startTime;
var timer = null;

var x = d3.scaleLinear()
    .domain([startTime,endTime])
    .range([0,targetValue])
    .clamp(true);

let step = function () {
    // increase time value
    currentValue = currentValue + (targetValue/endTime);
    // convert time value to time step
    time = x.invert(currentValue);
    // update the network
    update();
    // stop the timer
    if (currentValue >= targetValue) {
        timer.stop();
        currentValue = 0;
        bttn.text("Play")
        text.text(d => "T="+startTime);
        console.log("End of the timer");
    };
};

contains = ({start, end}, time) => start <= time && time < end

function update(){
    console.log("update Network");
    console.log(time);

    // Make copy to don't lose the data
    let copy = {...data};
    // TODO Instead of copy make a map to keep object properties
    network=copy;
    network.nodes = copy.nodes.filter(d => contains(d,time));
    network.edges = copy.edges.filter(d => contains(d,time));
    text.text(d => "T="+Math.round(time));
    render();
};

bttn.on('click', function() {
    if (bttn.text() == "Pause") {
        timer.stop();
        bttn.text("Play");
    }else{
        runTimer();
    };
  
});

function runTimer(){
    timer = d3.interval(step,duration);
    bttn.text("Pause");
}

runTimer();

// initialize timer
//let timer = d3.interval(step,duration);
