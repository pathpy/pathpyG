const scrubber = (values, {
  chartUpdate,
  format = value => value,
  initial = 0,
  delay = null,
  autoplay = true,
  loop = true,
  loopDelay = null,
  alternate = false
} = {}) => {
  values = Array.from(values);
  const form = d3.create('form')
    .style('font', '12px var(--sans-serif)')
    .style('font-variant-numeric', 'tabular-nums')
    .style('display', 'flex')
    .style('height', 33)
    .style('align-items', 'center')
    .attr('value', values[initial]);
  const formB = form.append('button')
    .attr('name', 'b')
    .attr('type', 'button')
    .style('margin-right', '0.4em')
    .style('width', '5em')
    .text('Play');

  const label = form.append('label')
    .style('display', 'flex')
    .style('align-items', 'center');
  const formI = label.append('input')
    .attr('name', 'i')
    .attr('type', 'range')
    .attr('min', 0)
    .attr('max', values.length - 1)
    .attr('value', initial)
    .attr('step', 1)
    .style('width', 180);
  const formO = label.append('output')
    .attr('name', 'o')
    .text(format(values[initial]));

  let frame = null;
  let timer = null;
  let interval = null;
  let direction = 1;

  const stop = () => {
    formB.text('Play');
    if (frame !== null) cancelAnimationFrame(frame), frame = null;
    if (timer !== null) clearTimeout(timer), timer = null;
    if (interval !== null) clearInterval(interval), interval = null;
  }

  const running = () => {
    return frame !== null || timer !== null || interval !== null;
  }

  const formIPostUpdate = (event) => {
    const index = parseInt(formI.property('value'));
    if (event && event.isTrusted && running()) stop();
    formO.property('value', format(values[index], index, values));

      chartUpdate(index);
  }

  const step = () => {
    formI.property('value', (parseInt(formI.property('value')) + direction + values.length) % values.length);
    formIPostUpdate();
  }

  const tick = () => {
    if (parseInt(formI.property('value')) === (direction > 0 ? values.length - 1 : direction < 0 ? 0 : NaN)) {
      if (!loop) return stop();
      if (alternate) direction = -direction;
      if (loopDelay !== null) {
        if (frame !== null) cancelAnimationFrame(frame), frame = null;
        if (interval !== null) clearInterval(interval), interval = null;
        timer = setTimeout(() => (step(), start()), loopDelay);
        return;
      }
    }
    if (delay === null) frame = requestAnimationFrame(tick);
    step();
  }

  const start = () => {
    formB.text('Pause');
    if (delay === null) frame = requestAnimationFrame(tick);
    else interval = setInterval(tick, delay);
  }

  formI.on('input', (event) => {
    formIPostUpdate(event);
  });
  formB.on('click', () => {
    if (running()) return stop();
    direction = alternate && parseInt(formI.property('value')) === values.length - 1 ? -1 : 1;
    formI.property('value', (parseInt(formI.property('value')) + direction) % values.length);
    formIPostUpdate();

    start();
  });

  if (autoplay) start();
  else stop();

  return form;
}


// Initialize Network
const network = Network(config);

// Function to filter elements
const contains = ({start,end}, time) => start <= time && time < end;

// Function to update network over time
const update = (index) => {
    const time = index;//times[index];
    const nodes = data.nodes.filter(d => contains(d, time));
    const links = data.edges.filter(d => contains(d, time));
    network.update({nodes,links});
};

// Range function
const range = (start, stop, step) =>
      Array.from({ length: (stop - start) / step + 1 }, (_, i) => start + i * step);

// Considered time range
times = range(d3.min(data.nodes, d => d.start),d3.max(data.nodes, d => d.end),1);

// Initiate Network
update(1);

// Add counter and start updating network
const scrubberForm = scrubber(times,{chartUpdate:update, delay: config.delta || 300});
d3.select(config.selector).append(() => scrubberForm.node());
