const shell = document.getElementById("viewer-shell");
const canvas = document.getElementById("stage");
const loading = document.getElementById("viewer-loading");
const progressInput = document.getElementById("progress");
const progressValue = document.getElementById("progress-value");
const playButton = document.getElementById("play");
const layerSelect = document.getElementById("layer");
const colorSelect = document.getElementById("color-mode");
const cameraSelect = document.getElementById("camera");
const resetButton = document.getElementById("reset-camera");
const infoPanel = document.getElementById("viewer-info");

const statusColors = {
  completed: "#0d8b87",
  current: "#ef6a32",
  next: "#e3a21a",
};

const layerGroups = {
  all: new Set(Object.keys(payload.typeLabels)),
  base: new Set(["bottom_slab"]),
  middle: new Set(["platform_slab", "column", "beam", "middle_slab"]),
  upper: new Set(["side_wall", "roof_slab", "stair"]),
};

const timelineEnd = Math.max(
  ...payload.components.map((component) => Number(component.end) || 0),
  1,
);
const cranePalette = ["#ef6a32", "#0d8b87", "#e3a21a", "#8a67c7"];
const craneColors = new Map(
  payload.cranes.map((crane, index) => [
    crane.id,
    cranePalette[index % cranePalette.length],
  ]),
);
let currentTime = 0;
let playTimer = null;
let selectedMesh = null;
let pointerMoved = false;

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
renderer.shadowMap.enabled = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color("#f4f1e8");
scene.fog = new THREE.Fog("#f4f1e8", payload.dimensions.spanX * 1.8, payload.dimensions.spanX * 3.8);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 5000);
const target = new THREE.Vector3(0, payload.dimensions.height * 0.36, 0);
const modelGroup = new THREE.Group();
scene.add(modelGroup);

scene.add(new THREE.HemisphereLight(0xffffff, 0x69736d, 1.55));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
keyLight.position.set(-80, 130, 90);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xcdd8d3, 0.8);
fillLight.position.set(90, 70, -80);
scene.add(fillLight);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(
    payload.dimensions.spanX * 1.25,
    payload.dimensions.spanY * 1.35,
  ),
  new THREE.MeshStandardMaterial({
    color: 0xd9d5ca,
    roughness: 0.98,
    metalness: 0.0,
  }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.22;
modelGroup.add(ground);

const gridSize = Math.max(payload.dimensions.spanX, payload.dimensions.spanY) * 1.3;
const grid = new THREE.GridHelper(gridSize, 24, 0xb7b8b0, 0xdad7cf);
grid.position.y = -0.18;
grid.scale.z = payload.dimensions.spanY / Math.max(payload.dimensions.spanX, 1);
modelGroup.add(grid);

const meshes = [];
const markers = [];
const edgeLines = [];
const gantryStates = [];
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function threePosition(x, y, z) {
  return new THREE.Vector3(
    x - payload.dimensions.centerX,
    z - payload.dimensions.minZ,
    y - payload.dimensions.centerY,
  );
}

function makeGeometry(component) {
  const positions = new Float32Array(component.vertices.length * 3);
  for (let index = 0; index < component.vertices.length; index += 1) {
    const vertex = component.vertices[index];
    positions[index * 3] = vertex[0] - payload.dimensions.centerX;
    positions[index * 3 + 1] = vertex[2] - payload.dimensions.minZ;
    positions[index * 3 + 2] = vertex[1] - payload.dimensions.centerY;
  }
  const indices = [];
  component.faces.forEach((face) => indices.push(face[0], face[1], face[2]));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function makeCraneLabel(text, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 384;
  canvas.height = 96;
  const context = canvas.getContext("2d");
  context.fillStyle = "rgba(24, 33, 31, 0.92)";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = color;
  context.lineWidth = 8;
  context.strokeRect(4, 4, canvas.width - 8, canvas.height - 8);
  context.fillStyle = "#fffdf7";
  context.font = "700 48px Microsoft YaHei, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    }),
  );
  sprite.scale.set(5.8, 1.45, 1);
  return sprite;
}

payload.components.forEach((component, index) => {
  const geometry = makeGeometry(component);
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(component.color),
    roughness: 0.84,
    metalness: 0.025,
    flatShading: true,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData = {
    index,
    component,
    order: component.order,
    type: component.type,
  };
  modelGroup.add(mesh);
  meshes.push(mesh);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry, 30),
    new THREE.LineBasicMaterial({
      color: 0x183a37,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    }),
  );
  edges.visible = false;
  modelGroup.add(edges);
  edgeLines.push(edges);

  const marker = new THREE.Mesh(
    new THREE.OctahedronGeometry(
      Math.max(0.16, Math.min(0.42, component.weight * 0.015)),
      0,
    ),
    new THREE.MeshBasicMaterial({
      color: new THREE.Color(component.color),
      transparent: true,
      opacity: 0.62,
      depthWrite: false,
    }),
  );
  marker.position.copy(threePosition(component.x, component.y, component.z));
  marker.userData = { index, component };
  modelGroup.add(marker);
  markers.push(marker);
});

payload.cranes.forEach((crane) => {
  const craneColor = craneColors.get(crane.id) || "#ef6a32";
  const railStart = threePosition(crane.railStartX, crane.trolleyMinY, 0.12);
  const railEnd = threePosition(crane.railEndX, crane.trolleyMaxY, 0.12);
  const railGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(railStart.x, railStart.y, railStart.z),
    new THREE.Vector3(railEnd.x, railStart.y, railStart.z),
  ]);
  modelGroup.add(
    new THREE.Line(
      railGeometry,
      new THREE.LineBasicMaterial({ color: 0x59655f }),
    ),
  );
  const farRailGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(railStart.x, railStart.y, railEnd.z),
    new THREE.Vector3(railEnd.x, railStart.y, railEnd.z),
  ]);
  modelGroup.add(
    new THREE.Line(
      farRailGeometry,
      new THREE.LineBasicMaterial({ color: 0x59655f }),
    ),
  );

  const span = crane.trolleyMaxY - crane.trolleyMinY;
  const gantryHeight = payload.dimensions.height + 3.4;
  const gantryGroup = new THREE.Group();
  gantryGroup.position.x =
    crane.x - payload.dimensions.centerX;
  const legMaterial = new THREE.MeshStandardMaterial({
    color: 0x3f4d48,
    roughness: 0.68,
    metalness: 0.20,
  });
  const legGeometry = new THREE.BoxGeometry(0.24, gantryHeight, 0.24);
  const frontLeg = new THREE.Mesh(legGeometry, legMaterial);
  frontLeg.position.set(
    0,
    gantryHeight * 0.5,
    crane.trolleyMinY - payload.dimensions.centerY,
  );
  const backLeg = frontLeg.clone();
  backLeg.position.z =
    crane.trolleyMaxY - payload.dimensions.centerY;
  gantryGroup.add(frontLeg, backLeg);

  const topBeam = new THREE.Mesh(
    new THREE.BoxGeometry(0.25, 0.25, span),
    new THREE.MeshStandardMaterial({
      color: new THREE.Color(craneColor),
      roughness: 0.62,
      metalness: 0.22,
    }),
  );
  topBeam.position.y = gantryHeight;
  gantryGroup.add(topBeam);

  const trolley = new THREE.Mesh(
    new THREE.BoxGeometry(0.72, 0.34, 0.72),
    new THREE.MeshStandardMaterial({
      color: new THREE.Color(craneColor),
      roughness: 0.58,
      metalness: 0.18,
    }),
  );
  trolley.position.set(
    0,
    gantryHeight - 0.24,
    crane.y - payload.dimensions.centerY,
  );
  gantryGroup.add(trolley);
  const craneLabel = makeCraneLabel(crane.id, craneColor);
  craneLabel.position.set(0, gantryHeight + 1.05, 0);
  gantryGroup.add(craneLabel);
  modelGroup.add(gantryGroup);
  gantryStates.push({
    crane,
    group: gantryGroup,
    trolley,
    currentX: crane.x,
    currentY: crane.y,
  });
});

const legendElement = document.getElementById("viewer-legend");

function legendItem(label, color) {
  return (
    '<span class="legend-item">' +
    `<i class="legend-dot" style="background:${color}"></i>` +
    `${label}</span>`
  );
}

function renderLegend() {
  if (!legendElement) return;
  const mode = colorSelect.value;
  const items = [];
  if (mode === "type") {
    const usedTypes = [...new Set(payload.components.map((item) => item.type))];
    usedTypes.forEach((type) => {
      items.push(
        legendItem(
          payload.typeLabels[type] || type,
          payload.typeColors[type] || "#9da39f",
        ),
      );
    });
  } else if (mode === "risk") {
    items.push(
      legendItem("低风险 <20", "#7f9e3a"),
      legendItem("中风险 20-35", "#e3a21a"),
      legendItem("较高风险 35-50", "#ef6a32"),
      legendItem("高风险 ≥50", "#c74436"),
    );
  } else if (mode === "crane") {
    payload.cranes.forEach((crane) => {
      items.push(
        legendItem(
          `${crane.id} 吊装设备`,
          craneColors.get(crane.id) || "#9da39f",
        ),
      );
    });
  } else {
    items.push(
      legendItem("已完成", statusColors.completed),
      legendItem("当前吊装", statusColors.current),
      legendItem("下一件", statusColors.next),
      legendItem("待安装定位", "#9da39f"),
    );
  }
  items.push(
    '<span class="legend-item">鼠标左键旋转，滚轮缩放，点击构件查看属性</span>',
  );
  legendElement.innerHTML = items.join("");
}

function riskColor(risk) {
  if (risk >= 50) return "#c74436";
  if (risk >= 35) return "#ef6a32";
  if (risk >= 20) return "#e3a21a";
  return "#7f9e3a";
}

function componentColor(component, status) {
  const mode = colorSelect.value;
  if (mode === "crane") {
    return craneColors.get(component.crane) || "#9da39f";
  }
  if (mode === "risk") return riskColor(component.risk);
  if (mode === "type") return component.color;
  if (status === "current") return statusColors.current;
  if (status === "next") return statusColors.next;
  if (status === "completed") return statusColors.completed;
  return component.color;
}

function statusForComponent(component) {
  const start = Number(component.start) || 0;
  const end = Number(component.end) || start;
  const lookahead = Math.max(timelineEnd * 0.018, 1.5);
  if (currentTime >= end) return "completed";
  if (currentTime >= start) return "current";
  if (start <= currentTime + lookahead) return "next";
  return "pending";
}

function updateVisuals() {
  const progress = Number(progressInput.value);
  currentTime = (progress / 100) * timelineEnd;
  progressValue.textContent = `${progress}%`;
  const selectedLayer = layerSelect.value;
  const allowedTypes = layerGroups[selectedLayer];
  let visibleMeshCount = 0;
  let visibleMarkerCount = 0;

  meshes.forEach((mesh, index) => {
    const component = payload.components[index];
    const status = statusForComponent(component);
    const layerVisible = allowedTypes.has(component.type);
    const visible =
      progress > 0 &&
      currentTime >= (Number(component.start) || 0) &&
      layerVisible;
    mesh.visible = visible;
    markers[index].visible = layerVisible && !visible;
    if (visible) visibleMeshCount += 1;
    if (markers[index].visible) visibleMarkerCount += 1;

    if (visible) {
      const color = componentColor(component, status);
      mesh.material.color.set(color);
      mesh.material.emissive.set(
        status === "current"
          ? statusColors.current
          : status === "next"
            ? statusColors.next
            : "#000000",
      );
      mesh.material.emissiveIntensity =
        status === "current" ? 0.30 : status === "next" ? 0.14 : 0.0;
    }

    const recentCompleted =
      status === "completed" &&
      (Number(component.end) || 0) >=
        currentTime - Math.max(timelineEnd * 0.06, 1.5);
    const edgeVisible =
      visible &&
      (status === "current" || status === "next" || recentCompleted);
    edgeLines[index].visible = edgeVisible;
    if (edgeVisible) {
      edgeLines[index].material.color.set(
        status === "current"
          ? statusColors.current
          : status === "next"
            ? statusColors.next
            : "#0b6764",
      );
      edgeLines[index].material.opacity =
        status === "completed" ? 0.48 : 0.88;
    }
  });
  gantryStates.forEach((state) => {
    const craneTasks = payload.components
      .filter((component) => component.crane === state.crane.id)
      .sort((first, second) => first.start - second.start);
    const activeTask = craneTasks.find(
      (task) => task.start <= currentTime && task.end >= currentTime,
    );
    const upcomingTask = craneTasks.find((task) => task.start > currentTime);
    const completedTasks = craneTasks.filter((task) => task.end <= currentTime);
    const focusAssignment =
      progress > 0
        ? activeTask ||
          upcomingTask ||
          completedTasks[completedTasks.length - 1] ||
          null
        : null;
    state.currentX = focusAssignment ? focusAssignment.x : state.crane.x;
    state.currentY = focusAssignment ? focusAssignment.y : state.crane.y;
    state.group.position.x =
      state.currentX - payload.dimensions.centerX;
    state.trolley.position.z =
      state.currentY - payload.dimensions.centerY;
  });
  window.viewerStats = {
    progress,
    currentTime,
    timelineEnd,
    visibleMeshCount,
    visibleMarkerCount,
    totalComponents: payload.componentCount,
  };
  renderer.render(scene, camera);
}

function fitCamera(mode = cameraSelect.value) {
  const { spanX, spanY, height } = payload.dimensions;
  const radius = Math.max(spanX, spanY) * 0.82;
  camera.up.set(0, 1, 0);
  if (mode === "end") {
    camera.position.set(spanY * 1.72, height * 0.70, 0.01);
  } else if (mode === "side") {
    camera.position.set(
      0,
      height * 0.88,
      Math.max(spanY * 1.42, spanX * 0.58),
    );
  } else if (mode === "top") {
    camera.position.set(0, radius * 1.42, 0.01);
    camera.up.set(0, 0, -1);
  } else {
    camera.position.set(
      spanX * 0.54,
      height * 1.06,
      spanY * 0.78,
    );
  }
  camera.lookAt(target);
  renderer.render(scene, camera);
}

function resize() {
  const width = shell.clientWidth;
  const height = shell.clientHeight;
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);
}

let dragActive = false;
let previousX = 0;
let previousY = 0;
let dragDistance = 0;

canvas.addEventListener("pointerdown", (event) => {
  dragActive = true;
  previousX = event.clientX;
  previousY = event.clientY;
  dragDistance = 0;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragActive) return;
  const deltaX = event.clientX - previousX;
  const deltaY = event.clientY - previousY;
  previousX = event.clientX;
  previousY = event.clientY;
  dragDistance += Math.abs(deltaX) + Math.abs(deltaY);
  const offset = camera.position.clone().sub(target);
  const spherical = new THREE.Spherical().setFromVector3(offset);
  spherical.theta -= deltaX * 0.006;
  spherical.phi = Math.max(
    0.10,
    Math.min(Math.PI - 0.10, spherical.phi - deltaY * 0.006),
  );
  camera.position.copy(target).add(new THREE.Vector3().setFromSpherical(spherical));
  camera.lookAt(target);
  renderer.render(scene, camera);
});

canvas.addEventListener("pointerup", (event) => {
  dragActive = false;
  canvas.releasePointerCapture(event.pointerId);
  if (dragDistance < 5) selectComponent(event);
});

canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    const offset = camera.position.clone().sub(target);
    const distance = Math.max(8, offset.length() * (event.deltaY > 0 ? 1.08 : 0.92));
    camera.position.copy(target).add(offset.normalize().multiplyScalar(distance));
    camera.lookAt(target);
    renderer.render(scene, camera);
  },
  { passive: false },
);

function selectComponent(event) {
  const bounds = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
  pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const intersections = raycaster.intersectObjects(
    meshes.filter((mesh) => mesh.visible),
    false,
  );
  if (!intersections.length) {
    selectedMesh = null;
    infoPanel.innerHTML =
      "<strong>构件信息</strong><span>点击实体构件查看定位、吊装顺序和设备信息</span>";
    return;
  }
  selectedMesh = intersections[0].object;
  const component = selectedMesh.userData.component;
  infoPanel.innerHTML = `
    <strong>${component.id}</strong>
    <span>${component.typeLabel} · ${component.name}</span>
    <span>吊装顺序：${component.order}</span>
    <span>吊装设备：${component.crane || "待分配"}</span>
    <span>作业时间：${Number(component.start || 0).toFixed(1)} - ${Number(component.end || 0).toFixed(1)} min</span>
    <span>重量：${component.weight.toFixed(1)} t</span>
    <span>定位：${component.x.toFixed(2)}, ${component.y.toFixed(2)}, ${component.z.toFixed(2)} m</span>
    <span>综合风险：${component.risk.toFixed(1)}</span>
  `;
}

progressInput.addEventListener("input", updateVisuals);
layerSelect.addEventListener("change", updateVisuals);
colorSelect.addEventListener("change", () => {
  renderLegend();
  updateVisuals();
});
cameraSelect.addEventListener("change", () => fitCamera(cameraSelect.value));
resetButton.addEventListener("click", () => {
  cameraSelect.value = "iso";
  fitCamera("iso");
});

playButton.addEventListener("click", () => {
  if (playTimer) {
    window.clearInterval(playTimer);
    playTimer = null;
    playButton.textContent = "播放";
    return;
  }
  if (Number(progressInput.value) >= 100) progressInput.value = "0";
  playButton.textContent = "暂停";
  playTimer = window.setInterval(() => {
    progressInput.value = String(Math.min(100, Number(progressInput.value) + 1));
    updateVisuals();
    if (Number(progressInput.value) >= 100) {
      window.clearInterval(playTimer);
      playTimer = null;
      playButton.textContent = "播放";
    }
  }, 170);
});

new ResizeObserver(resize).observe(shell);
fitCamera("iso");
renderLegend();
updateVisuals();
resize();
loading.style.display = "none";
window.viewerReady = true;
