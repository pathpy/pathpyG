console.log("Temporal Network Template");
/* Resources
   https://bl.ocks.org/mapio/53fed7d84cd1812d6a6639ed7aa83868
   https://codepen.io/smlo/pen/JdMOej
   https://observablehq.com/@d3/temporal-force-directed-graph
*/

// console.log(data);

// variables from the config file
const selector = config.selector;
const width = config.width || 400;
const height = config.height || 400;
const delta = config.delta || 300;
const charge_distance = config.charge_distance || 400;
const charge_force = config.charge_force || -3000;

// variables for the temporal components
const startTime = config.start;
const endTime = config.end;
const targetValue = config.intervals || 300;
const duration = config.delta || 300;  


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

/*Time counter */
let text = svg.append("text")
    .text("T="+startTime)
    .attr("x", 20)
    .attr("y", 20)
    .attr("class", "labelText");

let bttn = svg.append("text")
    .attr("x",70)
    .attr("y", 20)
    .text("Play")
    .attr("class", "labelText");

/*Assign data to variable*/
let network = data

/*Render function to show dynamic networks*/
function render(){

    // get network data
    let nodeData = network.nodes;
    let edgeData = network.edges;

    // render network objects
    renderNodes(nodeData);
    renderEdges(edgeData);

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
        .style("r", d => d.size)
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
        .style("r", d => d.size)
        .style("fill", d => d.color)
        .style("opacity", d => d.opacity);
};

/*Render edge objects*/
function renderEdges(data){
    // console.log("render Edges")
    edges = container.select(".edges").selectAll(".link").data(data, d=> d.uid);

    let new_edges =  edges.enter().append("line")
        .attr("class", "link")
        .style("stroke", d => d.color)
        .style("stroke-opacity", d => d.opacity)
        .style("stroke-width", d => d.size);

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

/*OLD Simulation of the forces*/
// const simulation = d3.forceSimulation()
//       .force("charge", d3.forceManyBody().strength(-3000))
//       .force("center", d3.forceCenter(width / 2, height / 2))
//       .force("x", d3.forceX(width / 2).strength(1))
//       .force("y", d3.forceY(height / 2).strength(1))
//       .force("links", d3.forceLink()
//              .id( d => d.uid)
//              .distance(50).strength(1))
//       .on("tick", ticked);


/*Update of the node and edge objects*/
function ticked() {
    nodes.call(updateNodePositions);
    edges.call(updateEdgePositions);
};

/*Update link positions */
function updateEdgePositions(edges) {
    edges
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
};

/*Update node positions */
function updateNodePositions(nodes) {
    nodes
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
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
