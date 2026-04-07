/**
 * @license
 * Copyright 2016 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as nn from "./nn";

export class NetworkCanvasRenderer {
  private ctx: CanvasRenderingContext2D;
  private canvas: HTMLCanvasElement;
  private width: number;
  private height: number;

  // Red (Negative) and Blue (Positive) theme
  private colorPositive = "#0877bd"; // Blue
  private colorNegative = "#f5222d"; // Red
  private colorNeutral = "#e8eaeb";

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", {alpha: false});
    this.resize();
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = this.width * window.devicePixelRatio;
    this.canvas.height = this.height * window.devicePixelRatio;
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  render(network: nn.Node[][]) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const numLayers = network.length;
    const layerSpacing = this.width / (numLayers + 1);
    
    // Predetermine node positions
    const nodePositions: {x: number, y: number}[][] = [];
    
    network.forEach((layer, layerIdx) => {
      const x = (layerIdx + 1) * layerSpacing;
      const layerPositions: {x: number, y: number}[] = [];
      
      if (layerIdx === 0) {
        // Input Layer: 28x28 Grid
        const gridSize = 28;
        const cellSize = 5; // Fixed smaller cell size
        const startY = (this.height - (gridSize * cellSize)) / 2;
        const startX = x - (gridSize * cellSize) / 2;
        
        for (let i = 0; i < layer.length; i++) {
          const row = Math.floor(i / gridSize);
          const col = i % gridSize;
          layerPositions.push({
            x: startX + col * cellSize,
            y: startY + row * cellSize
          });
        }
      } else {
        // Hidden and Output Layers: Vertical Columns
        const numNodes = layer.length;
        const spacing = Math.min(30, (this.height - 40) / numNodes);
        const startY = (this.height - (numNodes * spacing)) / 2;
        
        for (let i = 0; i < numNodes; i++) {
          layerPositions.push({
            x: x,
            y: startY + i * spacing + spacing / 2
          });
        }
      }
      nodePositions.push(layerPositions);
    });

    // Draw Links (Weights) with High-Performance Batching
    const opacityBuckets = 10;
    const posPaths = new Array(opacityBuckets).fill(null).map(() => new Path2D());
    const negPaths = new Array(opacityBuckets).fill(null).map(() => new Path2D());

    for (let layerIdx = 1; layerIdx < numLayers; layerIdx++) {
      const currentLayer = network[layerIdx];
      const prevPositions = nodePositions[layerIdx - 1];
      const currentPositions = nodePositions[layerIdx];
      
      for (let i = 0; i < currentLayer.length; i++) {
        const node = currentLayer[i];
        const destPos = currentPositions[i];
        
        for (let j = 0; j < node.inputLinks.length; j++) {
            const link = node.inputLinks[j];
            const weight = link.weight;
            const absWeight = Math.abs(weight);
            
            // Performance Culling: Skip visually insignificant weights
            if (absWeight < 0.05) continue; 

            const srcPos = prevPositions[j];
            const opacity = Math.min(1, absWeight * 0.5);
            const bucketIdx = Math.min(opacityBuckets - 1, Math.floor(opacity * opacityBuckets));
            
            const path = weight > 0 ? posPaths[bucketIdx] : negPaths[bucketIdx];
            path.moveTo(srcPos.x, srcPos.y);
            path.lineTo(destPos.x, destPos.y);
        }
      }
    }

    // Execute Batch Draws
    ctx.lineWidth = 1;
    for (let b = 0; b < opacityBuckets; b++) {
        const alpha = (b + 1) / opacityBuckets;
        ctx.globalAlpha = alpha;
        
        // Positive Paths (Blue)
        ctx.strokeStyle = this.colorPositive;
        ctx.stroke(posPaths[b]);
        
        // Negative Paths (Red)
        ctx.strokeStyle = this.colorNegative;
        ctx.stroke(negPaths[b]);
    }
    ctx.globalAlpha = 1.0;

    // Draw Nodes (Activations)
    network.forEach((layer, layerIdx) => {
      const positions = nodePositions[layerIdx];
      const isInput = layerIdx === 0;
      const isOutput = layerIdx === network.length - 1;

      layer.forEach((node, nodeIdx) => {
        const pos = positions[nodeIdx];
        const activation = node.output;
        const absActivation = Math.abs(activation);
        
        ctx.globalAlpha = 1.0;
        ctx.shadowBlur = 0;

        if (isInput) {
            // Input pixels are small vibrant squares
            const size = 3;
            if (activation > 0.05) {
                ctx.fillStyle = `rgba(255, 255, 255, ${activation})`;
                if (activation > 0.5) {
                    ctx.shadowBlur = 4;
                    ctx.shadowColor = "white";
                }
                ctx.fillRect(pos.x - size/2, pos.y - size/2, size, size);
                ctx.shadowBlur = 0;
            }
        } else {
            // Neurons are circles
            const radius = isOutput ? 8 : 5;
            
            // Background dim state
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
            ctx.fill();

            if (absActivation > 0.05) {
                // Active state
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
                const color = activation > 0 ? this.colorPositive : this.colorNegative;
                ctx.fillStyle = color;
                ctx.globalAlpha = Math.min(1, absActivation);
                
                // Glow for high activation
                if (absActivation > 0.4) {
                    ctx.shadowBlur = 12;
                    ctx.shadowColor = color;
                }
                ctx.fill();
                ctx.shadowBlur = 0;
                ctx.globalAlpha = 1.0;

                // Bright core for very active nodes
                if (absActivation > 0.7) {
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, 2.5, 0, Math.PI * 2);
                    ctx.fillStyle = "white";
                    ctx.fill();
                }
            }
            
            // Node Border (very subtle)
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(255,255,255,0.1)";
            ctx.lineWidth = 1;
            ctx.stroke();
        }
      });
    });
  }
}
