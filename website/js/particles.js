/**
 * Full-Page Particle Background Effect using Three.js
 * Similar to Antigravity component but as a page background
 */

class ParticleBackground {
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.particles = [];
    this.mesh = null;

    // Configuration
    this.config = {
      count: 5000,
      magnetRadius: 120,
      ringRadius: 110,
      waveSpeed: 0.4,
      waveAmplitude: 1,
      particleSize: 2.5,
      lerpSpeed: 0.05,
      color: '#5227FF',
      autoAnimate: true,
      particleVariance: 1,
      depthFactor: 1,
      pulseSpeed: 3,
    };

    this.mousePos = { x: 0, y: 0 };
    this.virtualMouse = { x: 0, y: 0 };
    this.lastMouseMoveTime = Date.now();

    this.init();
    this.setupEventListeners();
    this.animate();
  }

  init() {
    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = null;

    // Camera setup - cover entire viewport
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 2000);
    this.camera.position.z = 200;

    // Renderer setup
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Create particles distributed across viewport
    this.createParticles();

    // Handle window resize
    window.addEventListener('resize', () => this.onWindowResize());
  }

  createParticles() {
    // Distribute particles across entire viewport with density falloff from center
    const width = window.innerWidth;
    const height = window.innerHeight;

    // Create capsule geometry manually using BoxGeometry as fallback
    const capsuleGeometry = new THREE.BoxGeometry(0.3, 1, 0.3);
    this.particleMaterial = new THREE.MeshBasicMaterial({
      color: this.config.color,
      transparent: true,
      opacity: 0.9,
    });

    // Create instanced mesh for better performance
    this.mesh = new THREE.InstancedMesh(capsuleGeometry, this.particleMaterial, this.config.count);
    this.scene.add(this.mesh);

    // Initialize particles with density falloff from center
    const dummy = new THREE.Object3D();
    for (let i = 0; i < this.config.count; i++) {
      // Use exponential distribution to compress towards center
      const randomAngle = Math.random() * Math.PI * 2;
      const randomRadius = Math.pow(Math.random(), 0.6) * (Math.max(width, height) / 2);

      const x = Math.cos(randomAngle) * randomRadius;
      const y = Math.sin(randomAngle) * randomRadius;
      const z = (Math.random() - 0.5) * 100;

      this.particles.push({
        t: Math.random() * 100,
        speed: 0.01 + Math.random() / 200,
        mx: x,
        my: y,
        mz: z,
        cx: x,
        cy: y,
        cz: z,
        randomRadiusOffset: (Math.random() - 0.5) * 2,
      });

      // Set initial position
      dummy.position.set(x, y, z);
      dummy.updateMatrix();
      this.mesh.setMatrixAt(i, dummy.matrix);
    }

    this.mesh.instanceMatrix.needsUpdate = true;
  }

  setupEventListeners() {
    document.addEventListener('mousemove', (e) => {
      // Convert screen coordinates to world space
      const width = window.innerWidth;
      const height = window.innerHeight;
      this.mousePos.x = (e.clientX / width) * (width / 2) - width / 4;
      this.mousePos.y = -(e.clientY / height) * (height / 2) + height / 4;
      this.lastMouseMoveTime = Date.now();
    });
  }

  onWindowResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate = () => {
    requestAnimationFrame(this.animate);

    const now = Date.now();
    const timeSinceLastMove = now - this.lastMouseMoveTime;

    // Auto-animate if no mouse movement for 3 seconds
    let targetX = this.mousePos.x;
    let targetY = this.mousePos.y;

    if (this.config.autoAnimate && timeSinceLastMove > 3000) {
      const time = now * 0.0003;
      targetX = Math.sin(time) * 80;
      targetY = Math.cos(time * 0.7) * 60;
    }

    // Smooth mouse tracking
    const smoothFactor = 0.04;
    this.virtualMouse.x += (targetX - this.virtualMouse.x) * smoothFactor;
    this.virtualMouse.y += (targetY - this.virtualMouse.y) * smoothFactor;

    const dummy = new THREE.Object3D();

    this.particles.forEach((particle, i) => {
      let { t, speed, mx, my, mz } = particle;
      t = particle.t += speed / 2;

      const dx = mx - this.virtualMouse.x;
      const dy = my - this.virtualMouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      let targetPos = { x: mx, y: my, z: mz };

      if (dist < this.config.magnetRadius) {
        const angle = Math.atan2(dy, dx);

        const wave = Math.sin(t * this.config.waveSpeed + angle) *
                     (0.5 * this.config.waveAmplitude);
        const deviation = particle.randomRadiusOffset * 5;

        const currentRingRadius = this.config.ringRadius + wave + deviation;

        targetPos.x = this.virtualMouse.x + currentRingRadius * Math.cos(angle);
        targetPos.y = this.virtualMouse.y + currentRingRadius * Math.sin(angle);
        targetPos.z = mz + Math.sin(t) * (1 * this.config.waveAmplitude);
      }

      particle.cx += (targetPos.x - particle.cx) * this.config.lerpSpeed;
      particle.cy += (targetPos.y - particle.cy) * this.config.lerpSpeed;
      particle.cz += (targetPos.z - particle.cz) * this.config.lerpSpeed;

      // Update instance matrix with rotation
      dummy.position.set(particle.cx, particle.cy, particle.cz);
      dummy.rotation.x = t * 0.5;
      dummy.rotation.y = t * 0.3;
      dummy.rotation.z = t * 0.7;
      dummy.updateMatrix();
      this.mesh.setMatrixAt(i, dummy.matrix);
    });

    this.mesh.instanceMatrix.needsUpdate = true;

    this.renderer.render(this.scene, this.camera);
  };
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('particleCanvas');
  if (canvas) {
    new ParticleBackground(canvas);
  }
});
