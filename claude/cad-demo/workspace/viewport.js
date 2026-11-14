import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';

const loader = new STLLoader();
const orientations = {
  iso: [1.05, -1.48, 1.22],
  top: [0, -.001, 1],
  bottom: [0, -.001, -1],
  front: [0, -1, 0],
  back: [0, 1, 0],
  left: [-1, 0, 0],
  right: [1, 0, 0],
};

async function binary(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Геометрия недоступна.');
  return response.arrayBuffer();
}

function makePoints(source, limit, material) {
  const positions = source.getAttribute('position');
  const step = Math.max(1, Math.ceil(positions.count / limit));
  const cloud = new Float32Array(Math.ceil(positions.count / step) * 3);
  for (let i = 0, j = 0; i < positions.count; i += step, j += 3) {
    cloud[j] = positions.getX(i);
    cloud[j + 1] = positions.getY(i);
    cloud[j + 2] = positions.getZ(i);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(cloud, 3));
  return new THREE.Points(geometry, material);
}

const green = new THREE.Color('#53dfd2');
const amber = new THREE.Color('#ffd07b');
const red = new THREE.Color('#f1836e');

function distanceColor(color, distance, tolerance) {
  const scale = distance / tolerance;
  return color.copy(scale < 1 ? green : amber)
    .lerp(scale < 1 ? amber : red, Math.min(1, scale < 1 ? scale : scale - 1));
}

function modelColors(values, vertexCount, tolerance, legacy = false) {
  if (values.length * 3 !== vertexCount) throw new Error('Карта отклонений не соответствует модели.');
  const compare = new Uint8Array(vertexCount * 3);
  const deviation = legacy ? null : new Uint8Array(compare.length);
  const amberShade = new THREE.Color('#edac60').toArray().map((value) => Math.round(value * 255));
  const white = [255, 255, 255];
  const sample = new THREE.Color();
  const rgb = [0, 0, 0];
  for (let face = 0; face < values.length; face++) {
    const outside = legacy ? values[face] : values[face] > tolerance;
    if (deviation && values[face] >= 2 * tolerance) {
      // так далеко от скана поверхность модели лежит там, где скана нет: достроенный скол или
      // неотсканированное место. Это не ошибка модели, красить её «красным» нельзя.
      rgb.splice(0, 3, ...amberShade);
    } else if (deviation) {
      distanceColor(sample, values[face], tolerance).toArray(rgb);
      for (let channel = 0; channel < 3; channel++) rgb[channel] = Math.round(rgb[channel] * 255);
    }
    for (let i = face * 9; i < face * 9 + 9; i += 3) {
      compare.set(outside ? amberShade : white, i);
      if (deviation) deviation.set(rgb, i);
    }
  }
  return {
    compare: new THREE.BufferAttribute(compare, 3, true),
    deviation: deviation ? new THREE.BufferAttribute(deviation, 3, true) : null,
  };
}

function makeDeviations(buffer, tolerance, material) {
  const values = new Float32Array(buffer);
  if (!values.length || values.length % 4) throw new Error('Проба отклонений повреждена.');
  const positions = new Float32Array(values.length / 4 * 3);
  const colors = new Float32Array(positions.length);
  const sample = new THREE.Color();
  for (let i = 0, j = 0; i < values.length; i += 4, j += 3) {
    positions.set(values.subarray(i, i + 3), j);
    distanceColor(sample, values[i + 3], tolerance).toArray(colors, j);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return new THREE.Points(geometry, material);
}

export class Viewport {
  constructor(element, marker) {
    this.element = element;
    this.marker = marker;
    this.axis = Array.from('XYZ', (name, index) => {
      const element = document.createElement('span');
      const line = document.createElement('i');
      const label = document.createElement('b');
      element.className = 'axis-axis axis-' + name.toLowerCase();
      label.textContent = name;
      element.append(line, label);
      document.getElementById('axis').append(element);
      const vector = new THREE.Vector3().setComponent(index, 1);
      return {vector, line, label};
    });
    this.inverseRotation = new THREE.Quaternion();
    this.axisVector = new THREE.Vector3();
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-80, 80, 80, -80, .01, 5000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(...orientations.iso);
    this.renderer = new THREE.WebGLRenderer({alpha: true, antialias: true});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.localClippingEnabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.06;
    this.renderer.domElement.setAttribute('aria-label', 'Интерактивная 3D-модель');
    element.prepend(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = .10;
    this.controls.minZoom = .3;
    this.controls.maxZoom = 18;
    this.section = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this.material = {
      model: new THREE.MeshStandardMaterial({color: 0xb5c3cc, metalness: .29, roughness: .44, side: THREE.DoubleSide}),
      scan: new THREE.MeshStandardMaterial({color: 0x76b6bc, metalness: .10, roughness: .72, side: THREE.DoubleSide}),
      ghost: new THREE.PointsMaterial({color: 0x83e6dc, transparent: true, opacity: .54,
        size: 1.45, sizeAttenuation: false, depthTest: true, depthWrite: false}),
      heatmap: new THREE.MeshBasicMaterial({vertexColors: true, toneMapped: false, side: THREE.DoubleSide}),
      deviation: new THREE.PointsMaterial({vertexColors: true, size: 3.6, toneMapped: false,
        sizeAttenuation: false, depthTest: true, depthWrite: false}),
    };
    this.scene.add(new THREE.HemisphereLight(0xe3f5ff, 0x55677e, 1.5));
    for (const [color, intensity, position] of [
      [0xfff4e4, 2.7, [40, -60, 105]], [0xa6d9ff, 1.6, [-65, 60, 35]], [0x79b8d5, 1.1, [70, 60, 5]],
    ]) {
      const light = new THREE.DirectionalLight(color, intensity);
      light.position.set(...position);
      this.scene.add(light);
    }
    this.objects = {};
    this.bounds = new THREE.Box3();
    this.center = new THREE.Vector3();
    this.size = 90;
    this.anchor = null;
    this.version = 0;
    this.mode = 'compare';
    this.direction = 'model';
    this.colors = {};
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(element);
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.drawMarker();
      this.drawAxes();
      this.render();
    });
  }

  resize() {
    const width = this.element.clientWidth;
    const height = this.element.clientHeight;
    if (!width || !height) return;
    this.stacked = this.mode === 'split' && width <= 500;
    const split = this.mode === 'split';
    const paneWidth = split && !this.stacked ? Math.floor(width / 2) : width;
    const paneHeight = split && this.stacked ? Math.floor(height / 2) : height;
    this.display = {width, height, paneWidth, paneHeight};
    const aspect = paneWidth / paneHeight;
    const vertical = this.size * .63 / Math.min(aspect, 1);
    Object.assign(this.camera, {left: -vertical * aspect, right: vertical * aspect, top: vertical, bottom: -vertical});
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  render() {
    const display = this.display;
    if (!display) return;
    const {width, height, paneWidth, paneHeight} = display;
    const {renderer, scene, camera} = this;
    if (this.mode !== 'split' || !this.objects.model) {
      renderer.setViewport(0, 0, width, height);
      renderer.render(scene, camera);
      return;
    }
    renderer.setScissorTest(true);
    for (const [layer, x, y] of [
      [1, 0, this.stacked ? height - paneHeight : 0],
      [2, this.stacked ? 0 : width - paneWidth, 0],
    ]) {
      camera.layers.set(0);
      camera.layers.enable(layer);
      renderer.setViewport(x, y, paneWidth, paneHeight);
      renderer.setScissor(x, y, paneWidth, paneHeight);
      renderer.render(scene, camera);
    }
    renderer.setScissorTest(false);
    camera.layers.enableAll();
  }

  async load(assets, tolerance) {
    const version = ++this.version;
    const [scanData, modelData] = await Promise.all([
      binary(assets.scan), assets.model ? binary(assets.model) : null,
    ]);
    if (version !== this.version) return;
    const scan = loader.parse(scanData);
    const model = modelData ? loader.parse(modelData) : null;
    for (const object of Object.values(this.objects)) {
      this.scene.remove(object);
      object.geometry.dispose();
    }
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
    }
    this.objects = {scan: new THREE.Mesh(scan, this.material.scan)};
    this.objects.scan.layers.set(1);
    this.colors = {};
    if (model) {
      this.objects.model = new THREE.Mesh(model, this.material.model);
      this.objects.model.layers.set(2);
      this.objects.ghost = makePoints(scan, 31000, this.material.ghost);
      this.objects.ghost.renderOrder = 2;
    }
    this.scene.add(...Object.values(this.objects));
    this.bounds.setFromObject(this.objects.model || this.objects.scan);
    this.bounds.getCenter(this.center);
    this.size = Math.max(1e-5, this.bounds.getSize(new THREE.Vector3()).length());
    this.grid = new THREE.GridHelper(this.size * 2.7, 32, 0x637f8e, 0x3a5664);
    this.grid.rotateX(Math.PI / 2);
    this.grid.position.set(this.center.x, this.center.y, this.bounds.min.z - this.size * .085);
    this.grid.material.transparent = true;
    this.grid.material.opacity = .28;
    this.scene.add(this.grid);
    this.anchor = null;
    this.clip(false, 53);
    this.setMode(this.mode);
    this.view('iso');
    if (model && (assets.model_deviation || assets.unobserved)) {
      const data = await binary(assets.model_deviation || assets.unobserved);
      if (version !== this.version) return;
      const legacy = !assets.model_deviation;
      const values = legacy ? new Uint8Array(data) : new Float32Array(data);
      this.colors = modelColors(values, model.getAttribute('position').count, tolerance, legacy);
      this.setMode(this.mode);
    }
    if (assets.deviations) {
      const buffer = await binary(assets.deviations);
      if (version !== this.version) return;
      this.objects.deviation = makeDeviations(buffer, tolerance, this.material.deviation);
      this.objects.deviation.renderOrder = 3;
      this.scene.add(this.objects.deviation);
      this.setMode(this.mode);
    }
  }

  setMode(mode, direction = this.direction) {
    const layoutChanged = (this.mode === 'split') !== (mode === 'split');
    this.mode = mode;
    this.direction = direction;
    if (layoutChanged) this.resize();
    this.camera.layers.enableAll();
    const {scan, model, ghost, deviation} = this.objects;
    const heatmap = mode === 'deviation' && direction === 'model' && Boolean(this.colors.deviation);
    if (scan) scan.visible = mode === 'scan' || mode === 'split' || !model;
    if (ghost) ghost.visible = mode === 'compare';
    if (deviation) deviation.visible = mode === 'deviation' && direction === 'scan';
    if (!model) return;
    model.visible = mode !== 'scan';
    model.material = heatmap ? this.material.heatmap : this.material.model;
    const colors = heatmap ? this.colors.deviation : mode === 'compare' ? this.colors.compare : null;
    if (colors) model.geometry.setAttribute('color', colors);
    const material = this.material.model;
    const shaded = mode === 'compare' && Boolean(colors);
    if (material.vertexColors !== shaded) {
      material.vertexColors = shaded;
      material.needsUpdate = true;
    }
    material.color.set(mode === 'deviation' ? 0x647484 : 0xb5c3cc);
  }

  view(name) {
    const direction = new THREE.Vector3(...orientations[name]).normalize();
    this.controls.target.copy(this.center);
    this.camera.position.copy(this.center).addScaledVector(direction, this.size * 2.1);
    this.camera.near = this.size * .004;
    this.camera.far = this.size * 20;
    this.camera.zoom = 1.38;
    this.resize();
    this.controls.update();
  }

  clip(enabled, position) {
    this.section.constant = this.bounds.min.z + (this.bounds.max.z - this.bounds.min.z) * position / 100;
    for (const material of Object.values(this.material)) {
      material.clippingPlanes = enabled ? [this.section] : [];
    }
    return this.section.constant;
  }

  focus(point) {
    this.anchor = new THREE.Vector3(...point);
    this.camera.position.add(this.anchor.clone().sub(this.controls.target));
    this.controls.target.copy(this.anchor);
    this.camera.zoom = 1.76;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  clearFocus() {
    this.anchor = null;
    this.marker.hidden = true;
  }

  drawAxes() {
    this.inverseRotation.copy(this.camera.quaternion).invert();
    for (const {vector, line, label} of this.axis) {
      const p = this.axisVector.copy(vector).applyQuaternion(this.inverseRotation);
      const x = p.x * 15, y = -p.y * 15;
      line.style.width = Math.hypot(x, y) + 'px';
      line.style.transform = 'rotate(' + Math.atan2(y, x) + 'rad)';
      label.style.left = (22 + x * 1.35) + 'px';
      label.style.top = (21 + y * 1.35) + 'px';
      label.style.opacity = Math.hypot(p.x, p.y) < .08 ? '0' : '1';
    }
  }

  drawMarker() {
    if (!this.anchor || this.mode === 'scan') {
      this.marker.hidden = true;
      return;
    }
    const p = this.anchor.clone().project(this.camera);
    this.marker.hidden = Math.abs(p.x) > 1 || Math.abs(p.y) > 1 || Math.abs(p.z) > 1;
    const split = this.mode === 'split';
    const x = (p.x + 1) * 50;
    const y = (1 - p.y) * 50;
    this.marker.style.left = (split && !this.stacked ? 50 + x / 2 : x) + '%';
    this.marker.style.top = (split && this.stacked ? 50 + y / 2 : y) + '%';
  }
}
