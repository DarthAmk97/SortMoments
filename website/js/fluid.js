/**
 * Simplified Fluid Simulation Background using Canvas
 * GPU-accelerated with WebGL
 */

class FluidSimulation {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', { alpha: true, antialias: true });

    if (!this.gl) {
      console.error('WebGL2 not supported, falling back to canvas');
      this.useFallback = true;
      return;
    }

    this.gl.clearColor(0, 0, 0, 0);
    this.gl.disable(this.gl.DEPTH_TEST);

    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.ratio = window.devicePixelRatio || 1;
    this.canvas.width = this.width * this.ratio;
    this.canvas.height = this.height * this.ratio;
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    // Simulation parameters
    this.gridScale = 1.5;
    this.gridWidth = Math.ceil(this.width / this.gridScale);
    this.gridHeight = Math.ceil(this.height / this.gridScale);

    // Mouse
    this.mouse = { x: 0, y: 0, px: 0, py: 0, vx: 0, vy: 0, down: false };
    this.mouseAutoX = 0;
    this.mouseAutoY = 0;
    this.lastMouseTime = Date.now();

    this.init();
    this.setupEventListeners();
    this.animate();
  }

  init() {
    // Shaders
    const vertShader = `#version 300 es
      precision highp float;
      in vec2 position;
      out vec2 uv;
      void main() {
        uv = position * 0.5 + 0.5;
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;

    const fragShader = `#version 300 es
      precision highp float;
      uniform sampler2D velocity;
      uniform vec2 aspectRatio;
      uniform vec3 color0;
      uniform vec3 color1;
      uniform vec3 color2;
      in vec2 uv;
      out vec4 outColor;

      void main() {
        vec2 vel = texture(velocity, uv).xy;
        float speed = length(vel) * 0.5;
        speed = clamp(speed, 0.0, 1.0);

        vec3 col;
        if (speed < 0.5) {
          col = mix(color0, color1, speed * 2.0);
        } else {
          col = mix(color1, color2, (speed - 0.5) * 2.0);
        }

        outColor = vec4(col, speed * 0.6);
      }
    `;

    const advectShader = `#version 300 es
      precision highp float;
      uniform sampler2D field;
      uniform sampler2D velocity;
      uniform float dt;
      uniform vec2 gridSize;
      in vec2 uv;
      out vec4 outColor;

      void main() {
        vec2 vel = texture(velocity, uv).xy * 2.0;
        vec2 coord = uv - vel * dt / gridSize;
        outColor = texture(field, coord);
      }
    `;

    const forceShader = `#version 300 es
      precision highp float;
      uniform vec2 mousePos;
      uniform vec2 mouseVel;
      uniform float radius;
      uniform vec2 gridSize;
      in vec2 uv;
      out vec4 outColor;

      void main() {
        vec2 delta = (uv - mousePos) * gridSize;
        float dist = length(delta);
        float falloff = smoothstep(radius, 0.0, dist);
        outColor = vec4(mouseVel * falloff * 50.0, 0.0, 1.0);
      }
    `;

    const divergenceShader = `#version 300 es
      precision highp float;
      uniform sampler2D velocity;
      uniform vec2 pixelSize;
      in vec2 uv;
      out vec4 outColor;

      void main() {
        float x0 = texture(velocity, uv - vec2(pixelSize.x, 0.0)).x;
        float x1 = texture(velocity, uv + vec2(pixelSize.x, 0.0)).x;
        float y0 = texture(velocity, uv - vec2(0.0, pixelSize.y)).y;
        float y1 = texture(velocity, uv + vec2(0.0, pixelSize.y)).y;
        float divergence = (x1 - x0 + y1 - y0) * 0.5;
        outColor = vec4(divergence, 0.0, 0.0, 1.0);
      }
    `;

    const pressureShader = `#version 300 es
      precision highp float;
      uniform sampler2D pressure;
      uniform sampler2D divergence;
      uniform vec2 pixelSize;
      in vec2 uv;
      out vec4 outColor;

      void main() {
        float p0 = texture(pressure, uv - vec2(pixelSize.x, 0.0)).x;
        float p1 = texture(pressure, uv + vec2(pixelSize.x, 0.0)).x;
        float p2 = texture(pressure, uv - vec2(0.0, pixelSize.y)).x;
        float p3 = texture(pressure, uv + vec2(0.0, pixelSize.y)).x;
        float div = texture(divergence, uv).x;
        float pressure = (p0 + p1 + p2 + p3) * 0.25 - div;
        outColor = vec4(pressure, 0.0, 0.0, 1.0);
      }
    `;

    const gradientShader = `#version 300 es
      precision highp float;
      uniform sampler2D velocity;
      uniform sampler2D pressure;
      uniform vec2 pixelSize;
      in vec2 uv;
      out vec4 outColor;

      void main() {
        float p0 = texture(pressure, uv - vec2(pixelSize.x, 0.0)).x;
        float p1 = texture(pressure, uv + vec2(pixelSize.x, 0.0)).x;
        float p2 = texture(pressure, uv - vec2(0.0, pixelSize.y)).x;
        float p3 = texture(pressure, uv + vec2(0.0, pixelSize.y)).x;
        vec2 vel = texture(velocity, uv).xy;
        vel -= vec2(p1 - p0, p3 - p2) * 0.5;
        outColor = vec4(vel, 0.0, 1.0);
      }
    `;

    this.programs = {
      display: this.createProgram(vertShader, fragShader),
      advect: this.createProgram(vertShader, advectShader),
      force: this.createProgram(vertShader, forceShader),
      divergence: this.createProgram(vertShader, divergenceShader),
      pressure: this.createProgram(vertShader, pressureShader),
      gradient: this.createProgram(vertShader, gradientShader)
    };

    // Create textures
    this.textures = {
      vel0: this.createTexture(this.gridWidth, this.gridHeight),
      vel1: this.createTexture(this.gridWidth, this.gridHeight),
      div: this.createTexture(this.gridWidth, this.gridHeight),
      pressure0: this.createTexture(this.gridWidth, this.gridHeight),
      pressure1: this.createTexture(this.gridWidth, this.gridHeight)
    };

    // Create framebuffers
    this.fbos = {
      vel0: this.createFBO(this.textures.vel0),
      vel1: this.createFBO(this.textures.vel1),
      div: this.createFBO(this.textures.div),
      pressure0: this.createFBO(this.textures.pressure0),
      pressure1: this.createFBO(this.textures.pressure1)
    };

    // Screen quad
    this.screenQuad = this.createScreenQuad();

    window.addEventListener('resize', () => this.onResize());
  }

  createProgram(vertSrc, fragSrc) {
    const prog = this.gl.createProgram();
    const vert = this.compileShader(this.gl.VERTEX_SHADER, vertSrc);
    const frag = this.compileShader(this.gl.FRAGMENT_SHADER, fragSrc);
    this.gl.attachShader(prog, vert);
    this.gl.attachShader(prog, frag);
    this.gl.linkProgram(prog);
    if (!this.gl.getProgramParameter(prog, this.gl.LINK_STATUS)) {
      console.error(this.gl.getProgramInfoLog(prog));
    }
    return prog;
  }

  compileShader(type, src) {
    const shader = this.gl.createShader(type);
    this.gl.shaderSource(shader, src);
    this.gl.compileShader(shader);
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error(this.gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  createTexture(w, h) {
    const tex = this.gl.createTexture();
    this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RG32F, w, h, 0, this.gl.RG, this.gl.FLOAT, null);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    return tex;
  }

  createFBO(tex) {
    const fbo = this.gl.createFramebuffer();
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fbo);
    this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, tex, 0);
    return fbo;
  }

  createScreenQuad() {
    const vao = this.gl.createVertexArray();
    const vbo = this.gl.createBuffer();
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    this.gl.bindVertexArray(vao);
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vbo);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
    const posLoc = this.gl.getAttribLocation(this.programs.display, 'position');
    this.gl.enableVertexAttribArray(posLoc);
    this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);
    return vao;
  }

  setupEventListeners() {
    document.addEventListener('mousemove', (e) => {
      this.mouse.px = this.mouse.x;
      this.mouse.py = this.mouse.y;
      this.mouse.x = (e.clientX / this.width) * 2 - 1;
      this.mouse.y = 1 - (e.clientY / this.height) * 2;
      this.mouse.vx = (this.mouse.x - this.mouse.px) * 0.5;
      this.mouse.vy = (this.mouse.y - this.mouse.py) * 0.5;
      this.lastMouseTime = Date.now();
    });
  }

  onResize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = this.width * this.ratio;
    this.canvas.height = this.height * this.ratio;
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  animate = () => {
    requestAnimationFrame(this.animate);

    const now = Date.now();
    const idleTime = now - this.lastMouseTime;

    // Auto-animate when idle
    if (idleTime > 2000) {
      const t = now * 0.0005;
      this.mouseAutoX = Math.sin(t) * 0.5;
      this.mouseAutoY = Math.cos(t * 0.7) * 0.3;
    } else {
      this.mouseAutoX = this.mouse.x;
      this.mouseAutoY = this.mouse.y;
    }

    // Simulation steps
    this.advect();
    this.addForce();
    this.computeDivergence();
    this.solvePressure();
    this.subtractPressure();
    this.display();
  };

  advect() {
    const prog = this.programs.advect;
    this.gl.useProgram(prog);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbos.vel1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.textures.vel0);
    const dt = 0.016;
    this.gl.uniform1f(this.gl.getUniformLocation(prog, 'dt'), dt);
    this.gl.uniform2f(this.gl.getUniformLocation(prog, 'gridSize'), this.gridWidth, this.gridHeight);
    this.gl.bindVertexArray(this.screenQuad);
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

    [this.textures.vel0, this.textures.vel1] = [this.textures.vel1, this.textures.vel0];
    [this.fbos.vel0, this.fbos.vel1] = [this.fbos.vel1, this.fbos.vel0];
  }

  addForce() {
    const prog = this.programs.force;
    this.gl.useProgram(prog);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbos.vel1);
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE);
    this.gl.uniform2f(this.gl.getUniformLocation(prog, 'mousePos'), this.mouseAutoX, this.mouseAutoY);
    this.gl.uniform2f(this.gl.getUniformLocation(prog, 'mouseVel'),
      this.mouse.vx * 2, this.mouse.vy * 2);
    this.gl.uniform1f(this.gl.getUniformLocation(prog, 'radius'), 0.15);
    this.gl.uniform2f(this.gl.getUniformLocation(prog, 'gridSize'), this.gridWidth, this.gridHeight);
    this.gl.bindVertexArray(this.screenQuad);
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    this.gl.disable(this.gl.BLEND);

    [this.textures.vel0, this.textures.vel1] = [this.textures.vel1, this.textures.vel0];
    [this.fbos.vel0, this.fbos.vel1] = [this.fbos.vel1, this.fbos.vel0];
  }

  computeDivergence() {
    const prog = this.programs.divergence;
    this.gl.useProgram(prog);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbos.div);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.textures.vel0);
    this.gl.uniform2f(this.gl.getUniformLocation(prog, 'pixelSize'),
      1 / this.gridWidth, 1 / this.gridHeight);
    this.gl.bindVertexArray(this.screenQuad);
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
  }

  solvePressure() {
    const prog = this.programs.pressure;
    this.gl.useProgram(prog);
    const iterations = 20;
    for (let i = 0; i < iterations; i++) {
      const fbo = i % 2 === 0 ? this.fbos.pressure1 : this.fbos.pressure0;
      const srcTex = i % 2 === 0 ? this.textures.pressure0 : this.textures.pressure1;
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fbo);
      this.gl.bindTexture(this.gl.TEXTURE_2D, srcTex);
      this.gl.uniform2f(this.gl.getUniformLocation(prog, 'pixelSize'),
        1 / this.gridWidth, 1 / this.gridHeight);
      this.gl.bindVertexArray(this.screenQuad);
      this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }
  }

  subtractPressure() {
    const prog = this.programs.gradient;
    this.gl.useProgram(prog);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.fbos.vel1);
    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.textures.vel0);
    this.gl.uniform1i(this.gl.getUniformLocation(prog, 'velocity'), 0);
    this.gl.activeTexture(this.gl.TEXTURE1);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.textures.pressure0);
    this.gl.uniform1i(this.gl.getUniformLocation(prog, 'pressure'), 1);
    this.gl.uniform2f(this.gl.getUniformLocation(prog, 'pixelSize'),
      1 / this.gridWidth, 1 / this.gridHeight);
    this.gl.bindVertexArray(this.screenQuad);
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

    [this.textures.vel0, this.textures.vel1] = [this.textures.vel1, this.textures.vel0];
    [this.fbos.vel0, this.fbos.vel1] = [this.fbos.vel1, this.fbos.vel0];
  }

  display() {
    const prog = this.programs.display;
    this.gl.useProgram(prog);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.textures.vel0);
    this.gl.uniform2f(this.gl.getUniformLocation(prog, 'aspectRatio'),
      this.width / this.height, 1);

    // Purple tones
    this.gl.uniform3f(this.gl.getUniformLocation(prog, 'color0'), 0.32, 0.16, 1.0);
    this.gl.uniform3f(this.gl.getUniformLocation(prog, 'color1'), 0.82, 0.36, 0.99);
    this.gl.uniform3f(this.gl.getUniformLocation(prog, 'color2'), 0.69, 0.59, 0.94);

    this.gl.bindVertexArray(this.screenQuad);
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('particleCanvas');
  if (canvas) {
    new FluidSimulation(canvas);
  }
});
