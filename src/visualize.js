function render(data, hotPoints, loops, matches) {
  const canvas = document.getElementById("canvas", { alpha: false });
  const ctx = canvas.getContext("2d");
  if (data.length === 0) {
    return;
  }

  let l = data[0].length;
  data.forEach((chunk, x) => {
    chunk.forEach((value, j) => {
      const y = l - j;
      const color = Math.floor(Math.min(Math.log2(value) * 10, 255));
      ctx.fillStyle = `rgb(${color}, ${color}, ${color})`;
      ctx.fillRect(x, y, 1, 1);
    });
  });

  ctx.fillStyle = `rgba(255, 0, 0, 0.5)`;
  hotPoints.forEach((offsets, x) => {
    offsets.forEach((j) => {
      const y = l - j;
      ctx.fillRect(x - 1, y - 1, 3, 3);
    });
  });

  ctx.fillStyle = `green`;
  const chunkSize = l * 2;
  loops.forEach((loop, i) => {
    const d = document.createElement("div");
    d.innerText = JSON.stringify(loop);
    d.style.marginLeft = `${loop.start / chunkSize}px`;
    d.style.width = `${(loop.end - loop.start) / chunkSize}px`;
    d.style.backgroundColor = `grey`;
    d.style.marginTop = "4px";
    const starts = matches.get(loop.delta);
    canvas.parentNode.appendChild(d);
    // Draw a histgram of matched "start" timestamps.
    if (starts != null) {
      const histogram = new Map();
      let histogramHeight = 10;
      starts.forEach(s => {
        const v = (histogram.get(s) ?? 0) + 1;
        histogram.set(s, v);
        if (v > histogramHeight) {
          histogramHeight = v;
        }
      });
      const histogramElement = document.createElement('canvas');
      histogramElement.width = data.length;
      histogramElement.height = histogramHeight;
      const ctx = histogramElement.getContext("2d");
      ctx.fillStyle = 'blue';
      histogram.forEach((h, x) => {
        ctx.fillRect(x, 0, 1, h);
      });
      canvas.parentNode.appendChild(histogramElement);
    }
  });
}
