const fineTunes = [];

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
  if (loops != null) {
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
        starts.forEach((s) => {
          const v = (histogram.get(s) ?? 0) + 1;
          histogram.set(s, v);
          if (v > histogramHeight) {
            histogramHeight = v;
          }
        });
        const histogramElement = document.createElement("canvas");
        histogramElement.width = data.length;
        histogramElement.height = histogramHeight;
        const ctx = histogramElement.getContext("2d");
        ctx.fillStyle = "blue";
        histogram.forEach((h, x) => {
          ctx.fillRect(x, 0, 1, h);
        });
        canvas.parentNode.appendChild(histogramElement);
      }
    });
  }

  fineTunes.forEach((fineTune) => {
    const { a, b, chunkSize, delta, confidence } = fineTune;
    const N = a.length / chunkSize;

    // Both a and b are Float32Array, with chunkSize * N length.
    // N is the number of "frame"s.
    // We want to render rows like:
    // - "Frame 1" (colspan=chunkSize)
    // - chunkSize columns of a[0] .. a[chunkSize-1].
    // - chunkSize columns of b[0] .. b[chunkSize-1].
    // - difference of a and b.
    // - "Frame 2" (colspan=chunkSize)
    // - chunkSize columns of a[chunkSize] .. a[chunkSize*2-1].
    // - chunkSize columns of b[chunkSize] .. b[chunkSize*2-1].
    // - difference of a and b.

    const domTable = document.createElement("table");
    domTable.className = "best-frames";
    let currentTr = null;
    const tr = (className) => {
      const tr = document.createElement("tr");
      if (className != null) {
        tr.className = className;
      }
      currentTr = tr;
      domTable.appendChild(tr);
      return tr;
    };
    const td = (text, colspan, parentTr = currentTr) => {
      const td = document.createElement("td");
      td.innerText = text;
      if (colspan != null) {
        td.colSpan = colspan;
      }
      if (parentTr != null) {
        parentTr.appendChild(td);
      }
      return td;
    };
    const coloredTd = (value, parentTr = currentTr) => {
      const s = value < 0.05 ? "" : value.toFixed(1);
      const e = td(s, null, parentTr);
      // value: 0 to 100.
      const v = Math.max(0, Math.min(255 - Math.floor(2.55 * value), 255));
      e.style.backgroundColor = `rgba(${v},${v},255)`;
      if (v < 128) {
        e.style.color = "white";
      }
    };
    const effectiveSize = chunkSize / 2;
    for (let i = 0; i < N; i++) {
      tr("title`");
      td(
        `Fine tune: delta ${delta} => confidence ${confidence.toFixed(3)}`,
        effectiveSize + 1
      );

      tr("row-frame");
      td(`Frame ${i + 1}`, effectiveSize + 1);

      const rowA = tr("row-a");
      td("a");

      const rowB = tr("row-b");
      td("b");

      const rowDelta = tr("row-delta");
      td("Δ");

      for (let j = 0; j < effectiveSize; j++) {
        const ai = a[i * chunkSize + j];
        const bi = b[i * chunkSize + j];
        coloredTd(ai, rowA);
        coloredTd(bi, rowB);
        coloredTd(Math.abs(ai - bi), rowDelta);
      }
    }

    document.body.appendChild(domTable);
  });
}
