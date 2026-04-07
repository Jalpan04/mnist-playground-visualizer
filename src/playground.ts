/* Copyright 2016 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as d3 from "d3";
import * as nn from "./nn";
import { State } from "./state";
import { MnistData } from "./mnist";
import { NetworkCanvasRenderer } from "./network_canvas";

/** Constants */
const INPUT_SIZE = 784;
const OUTPUT_SIZE = 10;

/** Global State */
let state = State.deserializeState();
let network: nn.Node[][] = [];
let mnistData: MnistData;
let networkCanvas: NetworkCanvasRenderer;
let iter = 0;
let isTraining = false;
let latestBatch: { xs: Float32Array, ys: Uint8Array } = null;

/** Prediction Chart State */
let predictionData: number[] = new Array(10).fill(0);

function makeGUI() {
  // Bind Play/Pause
  d3.select("#play-pause-button").on("click", () => {
    isTraining = !isTraining;
    d3.select("#play-pause-button").classed("is-playing", isTraining);
    d3.select("#play-pause-button i").text(isTraining ? "pause" : "play_arrow");
    if (isTraining) {
      trainLoop();
    }
  });

  // Bind Reset
  d3.select("#reset-button").on("click", () => {
    reset();
  });

  // Setup User Drawing Canvas
  setupUserDrawing();
  
  // Setup Prediction Bar Chart
  updatePredictionChart();
}

function setupUserDrawing() {
  const canvas = document.getElementById("user-draw-canvas") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d");
  const clearBtn = document.getElementById("clear-drawing");
  
  ctx.strokeStyle = "white";
  ctx.lineWidth = 12;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  let isDrawing = false;
  
  const getPos = (e: MouseEvent | TouchEvent) => {
    const rect = canvas.getBoundingClientRect();
    if (e instanceof MouseEvent) {
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    } else {
      const touch = (e as TouchEvent).touches[0];
      return { x: touch.clientX - rect.left, y: touch.clientY - rect.top };
    }
  };

  const start = (e: MouseEvent | TouchEvent) => {
    // Auto-pause training while drawing to focus on live prediction
    if (isTraining) {
        isTraining = false;
        d3.select("#play-pause-button").classed("is-playing", false);
        d3.select("#play-pause-button i").text("play_arrow");
    }

    isDrawing = true;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    if (e instanceof TouchEvent) e.preventDefault();
  };

  const draw = (e: MouseEvent | TouchEvent) => {
    if (!isDrawing) return;
    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    runInference();
    if (e instanceof TouchEvent) e.preventDefault();
  };

  const stop = () => {
    isDrawing = false;
  };

  canvas.addEventListener("mousedown", start);
  canvas.addEventListener("mousemove", draw);
  window.addEventListener("mouseup", stop);
  
  canvas.addEventListener("touchstart", start);
  canvas.addEventListener("touchmove", draw);
  canvas.addEventListener("touchend", stop);

  clearBtn.addEventListener("click", () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionData.fill(0);
    updatePredictionChart();
  });
}

function runInference() {
  const canvas = document.getElementById("user-draw-canvas") as HTMLCanvasElement;
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext("2d");
  
  // High quality downsampling
  tempCtx.drawImage(canvas, 0, 0, 28, 28);
  
  const imgData = tempCtx.getImageData(0, 0, 28, 28);
  const input = new Array(784);
  for (let i = 0; i < 784; i++) {
    input[i] = imgData.data[i * 4] / 255;
  }

  const output = nn.forwardProp(network, input);
  predictionData = output;
  updatePredictionChart();
  
  // Render the network in its new activated state
  networkCanvas.render(network);
}

function updatePredictionChart() {
  const container = d3.select("#prediction-bars");
  const bars = container.selectAll(".prediction-row")
    .data(predictionData);

  const maxVal = Math.max(...predictionData, 0.0001);

  const enter = bars.enter()
    .append("div")
    .attr("class", "prediction-row")
    .style("display", "flex")
    .style("align-items", "center")
    .style("gap", "10px")
    .style("margin-bottom", "4px");

  enter.append("span")
    .attr("class", "digit-label")
    .style("width", "15px")
    .style("font-weight", "bold")
    .style("color", "#888")
    .text((d, i) => i);

  const barBg = enter.append("div")
    .attr("class", "prediction-bar-bg")
    .style("flex", "1")
    .style("height", "12px")
    .style("background", "rgba(255,255,255,0.05)")
    .style("border-radius", "6px")
    .style("overflow", "hidden");

  barBg.append("div")
    .attr("class", "prediction-bar-fill")
    .style("height", "100%")
    .style("width", "0%")
    .style("transition", "width 0.2s ease-out, background 0.2s");

  enter.append("span")
    .attr("class", "digit-value")
    .style("width", "40px")
    .style("font-size", "11px")
    .style("color", "#aaa")
    .text("0%");

  // Updates
  container.selectAll(".prediction-bar-fill")
    .data(predictionData)
    .style("width", d => `${Math.max(2, (d / maxVal) * 100)}%`)
    .style("background", (d, i) => {
        const isMax = d === Math.max(...predictionData) && d > 0.1;
        return isMax ? "#0072ff" : "rgba(255,255,255,0.2)";
    });

  container.selectAll(".digit-value")
    .data(predictionData)
    .text(d => `${(d * 100).toFixed(0)}%`)
    .style("color", (d, i) => {
        return d === Math.max(...predictionData) && d > 0.1 ? "#fff" : "#aaa";
    });
}

async function reset() {
  iter = 0;
  isTraining = false;
  d3.select("#play-pause-button").classed("is-playing", false);
  d3.select("#play-pause-button i").text("play_arrow");
  d3.select("#iter-number").text("000,000");

  // Build Network
  const shape = [INPUT_SIZE].concat(state.networkShape).concat([OUTPUT_SIZE]);
  network = nn.buildNetwork(shape, state.activation, nn.Activations.SIGMOID,
      state.regularization, [], state.initZero);

  // Initialize Canvas Renderer
  const container = document.getElementById("network-canvas-container");
  if (!networkCanvas) {
    const canvas = document.createElement("canvas");
    container.appendChild(canvas);
    networkCanvas = new NetworkCanvasRenderer(canvas);
  }
  networkCanvas.resize();

  // Initialize Data
  if (!mnistData) {
    mnistData = new MnistData();
    await mnistData.load();
  }
  
  updateUI();
}

function updateUI() {
  const iterStr = iter.toLocaleString();
  let padded = iterStr;
  while(padded.length < 7) padded = "0" + padded;
  d3.select("#iter-number").text(padded);
  networkCanvas.render(network);
  
  // Update training preview
  if (latestBatch) {
    const previewCanvas = document.getElementById("train-preview-canvas") as HTMLCanvasElement;
    const ctx = previewCanvas.getContext("2d");
    const imgData = ctx.createImageData(28, 28);
    // Show last image in batch
    const offset = (state.batchSize - 1) * 784;
    for (let i = 0; i < 784; i++) {
        const val = latestBatch.xs[offset + i] * 255;
        imgData.data[i*4] = val;
        imgData.data[i*4+1] = val;
        imgData.data[i*4+2] = val;
        imgData.data[i*4+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }
}

function trainLoop() {
  if (!isTraining) return;
  oneStep();
  requestAnimationFrame(trainLoop);
}

function oneStep() {
  const batch = mnistData.getTrainBatch(state.batchSize);
  latestBatch = batch;
  
  for (let i = 0; i < state.batchSize; i++) {
    const x = Array.from(batch.xs.slice(i * 784, (i + 1) * 784));
    const y = Array.from(batch.ys.slice(i * 10, (i + 1) * 10));
    nn.forwardProp(network, x);
    nn.backProp(network, y, nn.Errors.SQUARE);
    nn.updateWeights(network, state.learningRate, state.regularizationRate);
  }

  iter++;
  if (iter % 1 === 0) {
    updateUI();
  }
}

// Start the app
reset().then(() => {
    makeGUI();
});
