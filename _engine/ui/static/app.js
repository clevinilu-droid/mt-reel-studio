(() => {
  const CHUNK_SIZE = 10 * 1024 * 1024;
  const STEP_SHORT = [
    ["s1", "B1", "Chuẩn hóa"],
    ["s2a", "B2A", "Nghe ASR"],
    ["s2b", "B2B", "Đọc OCR"],
    ["s3", "B3", "Khớp Sub"],
    ["s4", "B4", "Tách giọng"],
    ["s5", "B5", "Dịch"],
    ["s6", "B6", "Lồng tiếng"],
  ];

  let state = { jobs: [], logs: {}, telemetry: null, drive_ready: false, drive_info: "" };
  let selectedJobId = null;
  let activeTab = "work";
  let pickedLocalFile = null;
  let downloadsMap = {};
  let remoteTarget = localStorage.getItem("mt_remote_target") || "";
  let remoteStateTimer = null;

  const $ = (id) => document.getElementById(id);

  function apiFetch(url, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    if (remoteTarget) headers["X-Remote-Target"] = remoteTarget;
    return fetch(url, { ...opts, headers });
  }

  async function pullRemoteStateOnce() {
    if (!remoteTarget) return;
    try {
      const res = await apiFetch("/api/state");
      if (!res.ok) return;
      state = await res.json();
      $("wsDot").className = "dot";
      renderAll();
    } catch (_) {
      $("wsDot").className = "dot off";
    }
  }

  function setRemoteTarget(url, goTab = null) {
    remoteTarget = (url || "").trim();
    if (remoteTarget) localStorage.setItem("mt_remote_target", remoteTarget);
    else localStorage.removeItem("mt_remote_target");

    const sel = $("targetMachineSelect");
    if (sel) {
      sel.value = remoteTarget;
      sel.classList.toggle("remote-on", Boolean(remoteTarget));
    }
    if (remoteStateTimer) { clearInterval(remoteStateTimer); remoteStateTimer = null; }
    selectedJobId = null;
    if (remoteTarget) {
      pullRemoteStateOnce();
      remoteStateTimer = setInterval(pullRemoteStateOnce, 1200);
    } else {
      fetch("/api/state").then((r) => r.json()).then((d) => { state = d; renderAll(); }).catch(() => {});
    }
    if (activeTab === "add") { loadDownloads(); loadDriveProjects(); }
    if (goTab) switchTab(goTab);
  }

  function esc(s) {
    return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function statusText(st) {
    return {
      idle: "Sẵn sàng",
      pending: "Chờ",
      queued: "Trong hàng đợi",
      running: "Đang chạy",
      paused: "Tạm dừng",
      step1_done: "✓ Xong Bước 1",
      s3_done: "✓ Xong tới B3",
      done: "✓ Hoàn tất",
      error: "Lỗi",
    }[st] || st;
  }

  function badgeClass(st) {
    if (st === "done" || st === "step1_done" || st === "s3_done") return "b-done";
    if (st === "running" || st === "queued") return "b-run";
    if (st === "error") return "b-err";
    return "b-wait";
  }

  function switchTab(tab) {
    activeTab = tab;
    $("tabWork").classList.toggle("active", tab === "work");
    $("tabAdd").classList.toggle("active", tab === "add");
    $("tabList").classList.toggle("active", tab === "list");
    $("tabVast").classList.toggle("active", tab === "vast");
    $("viewWork").style.display = tab === "work" ? "block" : "none";
    $("viewAdd").style.display = tab === "add" ? "block" : "none";
    $("viewList").style.display = tab === "list" ? "block" : "none";
    $("viewVast").style.display = tab === "vast" ? "block" : "none";
    if (tab === "add") loadDownloads();
    if (tab === "vast") refreshVastInstances();
  }

  async function loadDriveProjects() {
    const sel = $("driveSelect");
    sel.innerHTML = '<option value="">Đang quét Google Drive...</option>';
    try {
      const res = await apiFetch("/api/drive/projects");
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      sel.innerHTML = '<option value="">-- Chọn gói phim trên Google Drive --</option>';
      for (const item of data.items || []) {
        const opt = document.createElement("option");
        opt.value = item.work_rel;
        opt.textContent = `${item.work_rel}  (${item.files.join(", ")})`;
        sel.appendChild(opt);
      }
      if (!(data.items || []).length) {
        sel.innerHTML = '<option value="">(Chưa có gói nào trong gdrive:MT-Video)</option>';
      }
    } catch (err) {
      sel.innerHTML = `<option value="">Lỗi Drive: ${err.message || err}</option>`;
    }
  }

  async function pullDriveJob() {
    const work_rel = $("driveSelect").value;
    if (!work_rel) {
      alert("Hãy bấm 🔄 để tải danh sách và chọn 1 gói phim trên Drive trước.");
      return;
    }
    const opts = readAddOptions("test_5min");
    $("btnPullDrive").disabled = true;
    $("btnPullDrive").textContent = "Đang kéo...";
    $("drivePullWrap").style.display = "block";
    $("drivePullLabel").textContent = `Đang kéo ${work_rel} từ Google Drive...`;
    $("drivePullBar").style.width = "10%";
    $("drivePullBar").classList.add("indeterminate");

    const t0 = Date.now();
    const pullTimer = setInterval(() => {
      const dp = state.drive_pull;
      if (dp && dp.status === "running" && dp.progress_pct > 0) {
        $("drivePullBar").classList.remove("indeterminate");
        $("drivePullBar").style.width = `${dp.progress_pct}%`;
        $("drivePullLabel").textContent = dp.detail || `Đang tải từ Drive: ${dp.progress_pct}%`;
      } else {
        const sec = Math.floor((Date.now() - t0) / 1000);
        $("drivePullLabel").textContent = `Đang kéo & giải nén ${work_rel} từ Google Drive... (${sec}s)`;
      }
    }, 500);

    try {
      const res = await apiFetch("/api/jobs/from-drive", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          work_rel,
          mode: "test_5min",
          langs: opts.langs,
          voice_mode: opts.voice_mode,
          auto_start: false,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      $("drivePullBar").classList.remove("indeterminate");
      $("drivePullBar").style.width = "100%";
      $("drivePullLabel").textContent = "✓ Đã kéo xong từ Google Drive!";
      selectedJobId = data.job.job_id;
      if (remoteTarget) await pullRemoteStateOnce();
      switchTab("work");
    } catch (err) {
      alert(`Kéo Drive thất bại: ${err.message || err}`);
    } finally {
      clearInterval(pullTimer);
      $("drivePullBar").classList.remove("indeterminate");
      $("drivePullWrap").style.display = "none";
      $("btnPullDrive").disabled = false;
      $("btnPullDrive").textContent = "⬇ Kéo Drive";
    }
  }

  async function loadDownloads() {
    try {
      const res = await apiFetch("/api/downloads");
      const data = await res.json();
      const sel = $("downloadSelect");
      const prev = sel.value;
      sel.innerHTML = '<option value="">-- Chọn file có sẵn trong Download/ --</option>';
      downloadsMap = {};
      for (const item of data.items || []) {
        downloadsMap[item.path] = item;
        const opt = document.createElement("option");
        opt.value = item.path;
        opt.textContent = `${item.work_rel}  (${item.size_mb} MB)`;
        sel.appendChild(opt);
      }
      if (prev && downloadsMap[prev]) sel.value = prev;
    } catch (_) {}
  }

  function renderTopbar() {
    const t = state.telemetry;
    if (!t) return;
    const g = t.gpu || {};
    const gpuStr = g.backend === "CUDA" && g.vram_total_gb != null
      ? `${g.name} ${g.util_pct}% (${g.vram_used_gb}/${g.vram_total_gb}GB)`
      : `${g.name || "GPU"}`;
    const drvStr = state.drive_ready ? "Drive ✓" : "Drive ✗";
    const tunStr = state.tunnel_url ? ` · Tunnel ✓` : "";
    $("sysInfo").textContent = `${gpuStr} · CPU ${t.cpu_pct}% · Trống ${t.disk_free_gb} GB · ${drvStr}${tunStr}`;
    $("jobCountBadge").textContent = String((state.jobs || []).length);
    const vb = $("vastKeyBadge");
    if (vb) {
      vb.className = `badge ${state.vast_configured ? "b-done" : "b-err"}`;
      vb.textContent = state.vast_configured ? "API Key ✓" : "Chưa có VAST_API_KEY";
    }
  }

  function renderWorkView() {
    const jobs = state.jobs || [];
    if (!jobs.length) {
      $("workEmpty").style.display = "block";
      $("workContent").style.display = "none";
      return;
    }
    if (!selectedJobId || !jobs.some((j) => j.job_id === selectedJobId)) {
      selectedJobId = jobs[0].job_id;
    }
    const job = jobs.find((j) => j.job_id === selectedJobId);
    $("workEmpty").style.display = "none";
    $("workContent").style.display = "block";

    $("wTitle").textContent = job.work_rel;
    $("wSubtitle").textContent = `Ngôn ngữ: ${(job.langs || []).join(", ").toUpperCase()}  ·  Ngoại ngữ: ${job.voice_mode === "clone_2" ? "Clone 1 Nam + 1 Nữ" : "Voice Design"}`;
    $("wBadge").textContent = statusText(job.status);
    $("wBadge").className = `badge ${badgeClass(job.status)}`;

    // 1. Stepper ngang 1 hàng siêu gọn
    $("stepper").innerHTML = STEP_SHORT.map(([k, code, name]) => {
      const st = (job.steps && job.steps[k]) || { status: "pending" };
      const icon = st.status === "done" ? "✓" : st.status === "running" ? "▶" : st.status === "error" ? "!" : "○";
      return `
        <div class="st-chip ${esc(st.status)}">
          <div class="st-num">${icon} ${code}</div>
          <div>${name}</div>
        </div>
      `;
    }).join("");

    // 2A. Hộp Tiến Trình LIVE: CHỈ hiện cho các task đang thực sự chạy/xếp hàng
    const isRunning = job.status === "running" || job.status === "queued";
    const ds = job.drive_sync || {};
    const isGhostLocalSync = Boolean(remoteTarget && String(ds.detail || "").includes("WAV -> FLAC"));
    const isDriveLive = !isGhostLocalSync && (ds.status === "running" || ds.status === "queued");
    const liveTasks = [];

    if (isRunning) {
      const curKey = job.current_step || "s1";
      const curSt = (job.steps && job.steps[curKey]) || { label: "Đang xử lý", progress_pct: 0, detail: "" };
      liveTasks.push({
        title: `▶ ${curSt.label}`,
        pct: Number(curSt.progress_pct || 0),
        detail: curSt.detail || "Đang khởi động...",
      });
      $("logDetails").open = true;
    }
    if (isDriveLive) {
      liveTasks.push({
        title: "☁️ Đang Upload Google Drive (chạy nền)",
        pct: Number(ds.progress_pct || 0),
        detail: ds.detail || "Đang đồng bộ...",
      });
    }

    const liveBox = $("liveTasksBox");
    if (liveTasks.length) {
      liveBox.style.display = "block";
      liveBox.innerHTML = liveTasks.map((t) => `
        <div class="live-item">
          <div class="live-head">
            <strong>${esc(t.title)}</strong>
            <span class="live-pct">${t.pct.toFixed(0)}%</span>
          </div>
          <div class="bar-wrap lg"><div class="bar-fill" style="width:${t.pct}%"></div></div>
          <div class="muted" style="margin-top:6px;">${esc(t.detail)}</div>
        </div>
      `).join("");
    } else {
      liveBox.style.display = "none";
      liveBox.innerHTML = "";
    }

    // 2B. Hộp Đã Hoàn Tất: tách riêng những bước & task Drive đã xong (không lẫn vào thanh live)
    const doneRows = [];
    for (const [k, code, name] of STEP_SHORT) {
      const st = (job.steps && job.steps[k]) || {};
      if (st.status === "done" && st.detail) {
        doneRows.push(`<div class="done-item"><strong>✓ ${code} — ${esc(name)}</strong><span>${esc(st.detail)}</span></div>`);
      } else if (st.status === "error") {
        doneRows.push(`<div class="done-item"><strong style="color:#fca5a5;">! ${code} — Lỗi</strong><span>${esc(st.detail || "")}</span></div>`);
      }
    }
    if (ds.status === "done") {
      doneRows.push(`<div class="done-item"><strong>☁️ Google Drive (Đã đồng bộ)</strong><span>${esc(ds.remote || ds.detail || "")}</span></div>`);
    } else if (ds.status === "error") {
      doneRows.push(`<div class="done-item"><strong style="color:#fca5a5;">☁️ Lỗi Upload Drive</strong><span>${esc(ds.detail || "")}</span></div>`);
    }

    const doneBox = $("completedSummaryBox");
    if (doneRows.length) {
      doneBox.style.display = "flex";
      doneBox.innerHTML = doneRows.join("");
    } else {
      doneBox.style.display = "none";
      doneBox.innerHTML = "";
    }

    // 3. ĐÚNG 1 NÚT CHÍNH TỰ ĐỘNG GỢI Ý THEO TRẠNG THÁI & PHẦN CỨNG
    const slot = $("primaryActionSlot");
    const hasCuda = state.telemetry?.gpu?.backend === "CUDA";
    if (isRunning) {
      slot.innerHTML = `<button type="button" class="btn amber lg" id="dynPrimary">⏸ Tạm dừng</button>`;
      $("dynPrimary").onclick = () => postJobAction("pause");
    } else if (hasCuda) {
      if (job.status === "done") {
        const driveDisabled = !state.drive_ready || isDriveLive;
        slot.innerHTML = `<button type="button" class="btn green lg" id="dynPrimary" ${driveDisabled ? "disabled" : ""}>📦 Upload Output lên Drive</button>`;
        $("dynPrimary").onclick = () => postJobAction("push-output");
      } else if (job.step1_test_ready && job.mode === "test_5min") {
        slot.innerHTML = `<button type="button" class="btn green lg" id="dynPrimary">🚀 Chạy Full (6 bước)</button>`;
        $("dynPrimary").onclick = () => postJobAction("start", { mode: "full", force_step1: false, target_stage: "all" });
      } else {
        slot.innerHTML = `<button type="button" class="btn primary lg" id="dynPrimary">🧪 Chạy Test 5 phút (6 bước)</button>`;
        $("dynPrimary").onclick = () => postJobAction("start", { mode: "test_5min", force_step1: false, target_stage: "all" });
      }
    } else if (job.step1_full_ready) {
      const driveDisabled = !state.drive_ready || ds.status === "running";
      slot.innerHTML = `<button type="button" class="btn green lg" id="dynPrimary" ${driveDisabled ? "disabled" : ""}>☁️ Upload Google Drive</button>`;
      $("dynPrimary").onclick = () => postJobAction("push-drive");
    } else if (job.step1_test_ready) {
      slot.innerHTML = `<button type="button" class="btn primary lg" id="dynPrimary">🚀 Chạy Full Bước 1</button>`;
      $("dynPrimary").onclick = () => postJobAction("start", { mode: "full", force_step1: false, target_stage: "s1" });
    } else {
      slot.innerHTML = `<button type="button" class="btn primary lg" id="dynPrimary">🧪 Chạy Test 5 phút</button>`;
      $("dynPrimary").onclick = () => postJobAction("start", { mode: "test_5min", force_step1: false, target_stage: "s1" });
    }

    // 4. Kết quả Khám bệnh Video Gốc
    const audit = job.audit_full || job.audit_test;
    if (audit) {
      $("auditCard").style.display = "block";
      $("auditModeLabel").textContent = job.audit_full ? "Bản Full phim" : "Bản Test 5 phút";
      const sp = audit.source_probe || {};
      const np = audit.normalized_probe || {};
      const tm = audit.timings_s || {};
      const sc = audit.spotcheck || {};
      $("auditMetrics").innerHTML = `
        <div class="metric"><div class="k">Thời gian Bước 1 (${esc(audit.encoder_used)})</div><div class="v">${esc(tm.total_step1_s)}s</div></div>
        <div class="metric"><div class="k">Sửa lỗi hình (VFR → CFR)</div><div class="v ok">${sp.is_vfr ? "Đã sửa VFR" : "Chuẩn"} → ${esc(np.width)}x${esc(np.height)} ${esc(np.avg_fps)}fps</div></div>
        <div class="metric"><div class="k">Khớp Phụ đề ↔ Tiếng nói</div><div class="v ok">${esc(sc.timeline_match_pct ?? "-")}% (${esc(sc.matched_ocr_lines ?? 0)}/${esc(sc.total_asr_lines ?? 0)})</div></div>
        <div class="metric"><div class="k">Độ lệch trung vị (Sub - Tiếng)</div><div class="v">${esc(sc.median_ocr_minus_asr_offset_s ?? "-")}s</div></div>
      `;
      const rows = [];
      for (const w of sc.windows || []) {
        for (const s of (w.samples || []).slice(0, 2)) {
          rows.push(`<tr><td>${esc(s.asr_t?.[0])}s</td><td>${esc(s.offset_s)}s</td><td>${esc(s.asr)}</td><td>${esc(s.ocr)}</td></tr>`);
        }
      }
      $("auditSamples").innerHTML = rows.length
        ? `<table class="samples"><thead><tr><th>Mốc</th><th>Lệch</th><th>Tiếng nghe (ASR)</th><th>Chữ trên hình (OCR)</th></tr></thead><tbody>${rows.join("")}</tbody></table>`
        : '<p class="muted">Không có mẫu câu.</p>';
    } else {
      $("auditCard").style.display = "none";
    }

    // 5. Log
    const logLines = (state.logs && state.logs[job.job_id]) || [];
    const logEl = $("logBox");
    logEl.textContent = logLines.length ? logLines.join("\n") : "(Chưa có nhật ký chạy mới)";
    logEl.scrollTop = logEl.scrollHeight;
  }

  function renderJobListView() {
    const jobs = state.jobs || [];
    const box = $("jobList");
    if (!jobs.length) {
      box.innerHTML = '<p class="muted">Chưa có phim nào.</p>';
      return;
    }
    box.innerHTML = jobs.map((j) => {
      const s1 = (j.steps && j.steps.s1) || {};
      return `
        <div class="job-row ${j.job_id === selectedJobId ? "active" : ""}" data-jid="${esc(j.job_id)}">
          <div>
            <div style="font-weight:600;">${esc(j.work_rel)}</div>
            <div class="muted">${esc(s1.detail || "Chưa chạy Bước 1")}</div>
          </div>
          <span class="badge ${badgeClass(j.status)}">${esc(statusText(j.status))}</span>
        </div>
      `;
    }).join("");

    box.querySelectorAll(".job-row").forEach((el) => {
      el.addEventListener("click", () => {
        selectedJobId = el.getAttribute("data-jid");
        switchTab("work");
        renderWorkView();
      });
    });
  }

  function renderAll() {
    renderTopbar();
    renderWorkView();
    renderJobListView();
  }

  function readAddOptions(mode) {
    const voice_mode = document.querySelector('input[name="voiceMode"]:checked')?.value || "clone_2";
    const langs = [...document.querySelectorAll(".langCk:checked")].map((el) => el.value);
    const work_rel = $("workRelInput").value.trim() || null;
    const auto_sync_drive = $("autoDriveCk").checked;
    return { mode, voice_mode, langs: langs.length ? langs : ["vi"], work_rel, auto_sync_drive };
  }

  function setPickedFile(file) {
    pickedLocalFile = file;
    if (file) {
      $("downloadSelect").value = "";
      $("pickedFile").style.display = "block";
      $("pickedFile").textContent = `Đã chọn: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    } else {
      $("pickedFile").style.display = "none";
      $("pickedFile").textContent = "";
    }
  }

  async function submitNewJob(mode) {
    const opts = readAddOptions(mode);
    const dlPath = $("downloadSelect").value;
    if (!pickedLocalFile && !dlPath) {
      alert("Hãy kéo thả / bấm Browse chọn video, hoặc chọn 1 file từ Download/.");
      return;
    }
    $("btnStartTest").disabled = true;
    $("btnStartFull").disabled = true;
    try {
      if (pickedLocalFile) {
        $("uploadWrap").style.display = "block";
        $("uploadLabel").textContent = `Chuẩn bị tải ${pickedLocalFile.name}...`;
        $("uploadBar").style.width = "0%";
        const initRes = await apiFetch("/api/upload/init", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: pickedLocalFile.name, size: pickedLocalFile.size }),
        });
        if (!initRes.ok) throw new Error(await initRes.text());
        const { upload_id, already_exists } = await initRes.json();

        if (!already_exists) {
          let offset = 0;
          while (offset < pickedLocalFile.size) {
            const end = Math.min(pickedLocalFile.size, offset + CHUNK_SIZE);
            const cRes = await apiFetch(`/api/upload/chunk?upload_id=${encodeURIComponent(upload_id)}`, {
              method: "POST",
              body: pickedLocalFile.slice(offset, end),
            });
            if (!cRes.ok) throw new Error(await cRes.text());
            offset = end;
            const pct = Math.round((offset / pickedLocalFile.size) * 100);
            $("uploadBar").style.width = `${pct}%`;
            $("uploadLabel").textContent = `Đang tải lên: ${pct}% (${(offset / 1024 / 1024).toFixed(0)}/${(pickedLocalFile.size / 1024 / 1024).toFixed(0)} MB)`;
          }
        }
        const finRes = await apiFetch("/api/upload/finish", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ upload_id, ...opts, auto_start: true }),
        });
        if (!finRes.ok) throw new Error(await finRes.text());
        const data = await finRes.json();
        selectedJobId = data.job.job_id;
        $("uploadWrap").style.display = "none";
        setPickedFile(null);
      } else {
        const res = await apiFetch("/api/jobs/from-path", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source_path: dlPath, ...opts, auto_start: true }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        selectedJobId = data.job.job_id;
      }
      if (remoteTarget) await pullRemoteStateOnce();
      switchTab("work");
    } catch (err) {
      alert(`Lỗi: ${err.message || err}`);
    } finally {
      $("btnStartTest").disabled = false;
      $("btnStartFull").disabled = false;
    }
  }

  async function postJobAction(endpoint, body = null) {
    if (!selectedJobId) return;
    $("moreMenu").open = false;
    const res = await apiFetch(`/api/jobs/${encodeURIComponent(selectedJobId)}/${endpoint}`, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) alert(`Thao tác thất bại: ${await res.text()}`);
    else if (remoteTarget) await pullRemoteStateOnce();
  }

  function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => { if (!remoteTarget) $("wsDot").className = "dot"; };
    ws.onmessage = (ev) => {
      if (remoteTarget) return;
      try {
        state = JSON.parse(ev.data);
        renderAll();
      } catch (_) {}
    };
    ws.onclose = () => {
      if (!remoteTarget) $("wsDot").className = "dot off";
      setTimeout(connectWS, 1500);
    };
  }

  async function searchVastOffers() {
    const wrap = $("vastOffersWrap");
    wrap.innerHTML = '<div class="muted">Đang truy vấn Vast.ai và lọc các máy đạt chuẩn nhanh nhất...</div>';
    $("btnSearchVast").disabled = true;
    try {
      const q = new URLSearchParams({
        gpu: $("vastGpu").value,
        sort_by: $("vastSort").value,
        min_inet: $("vastMinInet").value || "1000",
        max_dph: $("vastMaxDph").value || "0.65",
      });
      const res = await fetch(`/api/vast/offers?${q.toString()}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const offers = data.offers || [];
      if (!offers.length) {
        wrap.innerHTML = '<div class="muted">Không có máy nào đạt đủ tiêu chí lọc hiện tại. Thử tăng giá $/h tối đa hoặc giảm nhẹ min Mbps.</div>';
        return;
      }
      wrap.innerHTML = `
        <table class="vast-tbl">
          <thead>
            <tr>
              <th>GPU</th><th>VRAM</th><th>Down / Up</th><th>RAM / CPU</th><th>Disk BW</th><th>Vị trí</th><th>Giá</th><th></th>
            </tr>
          </thead>
          <tbody>
            ${offers.map((o) => `
              <tr>
                <td><strong>${esc(o.gpu_name)}</strong><div class="muted">PCIe ${esc(o.pcie_bw_gbs)} GB/s</div></td>
                <td>${esc(o.vram_gb)} GB<div class="muted">CUDA ${esc(o.cuda_max)}</div></td>
                <td>${esc(o.inet_down_mbps)} / ${esc(o.inet_up_mbps)} Mbps</td>
                <td>${esc(o.ram_gb)} GB / ${esc(o.cpu_cores)}c</td>
                <td>${esc(o.disk_bw_mbs)} MB/s</td>
                <td>${esc(o.geolocation)}<div class="muted">Rel ${esc(o.reliability_pct)}%</div></td>
                <td><strong>$${Number(o.dph_total).toFixed(3)}</strong>/h</td>
                <td><button type="button" class="btn primary sm" data-rent="${o.id}">Thuê máy</button></td>
              </tr>
            `).join("")}
          </tbody>
        </table>`;
      wrap.querySelectorAll("[data-rent]").forEach((btn) => {
        btn.onclick = () => rentVastOffer(Number(btn.dataset.rent));
      });
    } catch (err) {
      wrap.innerHTML = `<div class="muted" style="color:#fca5a5;">${esc(err.message || err)}</div>`;
    } finally {
      $("btnSearchVast").disabled = false;
    }
  }

  async function rentVastOffer(offerId) {
    const imgVal = $("vastImage").value.trim();
    if (!imgVal) {
      alert("Hãy nhập Docker Image (vd: ghcr.io/<user>/mt-reel:v2) hoặc Template Hash ID của MT Reel trước khi thuê.");
      $("vastImage").focus();
      return;
    }
    const looksLikeHash = !imgVal.includes("/") && !imgVal.includes(":");
    if (!confirm(`Xác nhận thuê Vast.ai offer #${offerId} (ổ cứng 120GB, tự mở port 7860 + tự nạp rclone)?`)) return;
    try {
      const res = await fetch("/api/vast/rent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          offer_id: offerId,
          image: looksLikeHash ? "" : imgVal,
          template_hash_id: looksLikeHash ? imgVal : "",
          disk_gb: 120,
          include_rclone: true,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      await refreshVastInstances();
      $("vastInstancesWrap").scrollIntoView({ behavior: "smooth", block: "nearest" });
      alert("Đã gửi lệnh thuê máy Vast.ai thành công! Xem trạng thái tải Docker & khởi động ngay ở hộp bên dưới.");
    } catch (err) {
      alert(`Thuê thất bại: ${err.message || err}`);
    }
  }

  let vastPollTimer = null;
  let vastOpenLogs = {};

  function vastStatusBadge(x) {
    const st = String(x.actual_status || "").toLowerCase();
    if (st === "running") return '<span class="badge b-done">🟢 Đang chạy (Sẵn sàng)</span>';
    if (st === "loading" || st === "creating" || st === "created") return '<span class="badge b-run">⏳ Đang tải & khởi tạo...</span>';
    return `<span class="badge b-wait">${esc(st || "chưa rõ")}</span>`;
  }

  async function loadVastLogs(instanceId) {
    vastOpenLogs[instanceId] = "Đang tải log từ Vast.ai...";
    const box = document.getElementById(`vastLog_${instanceId}`);
    if (box) { box.style.display = "block"; box.textContent = vastOpenLogs[instanceId]; }
    try {
      const res = await fetch(`/api/vast/instances/${instanceId}/logs`);
      if (!res.ok) throw new Error(await res.text());
      const d = await res.json();
      vastOpenLogs[instanceId] = (d.tunnel_url ? `Tunnel HTTPS: ${d.tunnel_url}\n\n` : "") + (d.logs || "(Chưa có log container — máy có thể vẫn đang giải nén Docker)");
      if (box) box.textContent = vastOpenLogs[instanceId];
    } catch (err) {
      vastOpenLogs[instanceId] = `Lỗi lấy log: ${err.message || err}`;
      if (box) box.textContent = vastOpenLogs[instanceId];
    }
  }

  function syncTargetMachineDropdown(list) {
    const sel = $("targetMachineSelect");
    if (!sel) return;
    sel.innerHTML = '<option value="">🖥️ Máy nhà (Local)</option>';
    const validUrls = new Set();
    for (const x of list || []) {
      const u = x.direct_ui_url || "";
      if (u) {
        validUrls.add(u);
        const opt = document.createElement("option");
        opt.value = u;
        opt.textContent = `⚡ #${x.id} ${x.gpu_name} (${x.actual_status})`;
        sel.appendChild(opt);
      }
    }
    if (remoteTarget && !validUrls.has(remoteTarget)) {
      setRemoteTarget("");
    } else {
      sel.value = remoteTarget;
      sel.classList.toggle("remote-on", Boolean(remoteTarget));
    }
  }

  async function refreshVastInstances() {
    const wrap = $("vastInstancesWrap");
    try {
      const res = await fetch("/api/vast/instances");
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (!data.configured) {
        syncTargetMachineDropdown([]);
        wrap.innerHTML = '<div class="muted">Chưa có VAST_API_KEY trong file .env trên máy local.</div>';
        return;
      }
      const list = data.instances || [];
      syncTargetMachineDropdown(list);
      if (!list.length) {
        wrap.innerHTML = '<div class="muted">Hiện không có máy Vast.ai nào đang thuê.</div>';
        return;
      }
      wrap.innerHTML = list.map((x) => {
        const uiLink = x.direct_ui_url || "";
        const stMsg = x.status_msg || (x.actual_status === "running" ? "Container đang chạy bình thường" : "Đang chờ Vast.ai phản hồi trạng thái...");
        const savedLog = vastOpenLogs[x.id] || "";
        const isCurrent = Boolean(uiLink && remoteTarget === uiLink);
        return `
          <div class="vast-card">
            <div class="vast-card-top">
              <div class="vast-card-title">
                <span>1x ${esc(x.gpu_name)}</span>
                ${vastStatusBadge(x)}
                <span class="muted">${esc(x.geolocation)} · Rel ${esc(x.reliability_pct)}%</span>
                ${x.public_ipaddr ? `<span class="badge b-wait">${esc(x.public_ipaddr)}</span>` : ""}
              </div>
              <div style="font-size:15px; font-weight:700; color:#86efac;">$${Number(x.dph_total || 0).toFixed(3)}/hr</div>
            </div>

            <div class="vast-card-grid">
              <div class="vast-kv">
                <div class="k">Instance / Host / Machine</div>
                <div class="v">ID: #${esc(x.id)}<br/>Host ${esc(x.host_id)} · M:${esc(x.machine_id)}<br/>VRAM ${esc(x.vram_used_gb)}/${esc(x.vram_total_gb)} GB (CUDA ${esc(x.cuda_max)})</div>
              </div>
              <div class="vast-kv">
                <div class="k">Mạng & PCIe</div>
                <div class="v">⬆ ${esc(x.inet_up_mbps)} Mbps<br/>⬇ ${esc(x.inet_down_mbps)} Mbps<br/>PCIe ${esc(x.pcie_bw_gbs)} GB/s · DLPerf ${esc(x.dlperf)}</div>
              </div>
              <div class="vast-kv">
                <div class="k">CPU & RAM</div>
                <div class="v">${esc(x.cpu_cores)} CPU<br/>${esc((x.cpu_name || "").slice(0, 26))}<br/>RAM ${esc(x.ram_used_gb)} / ${esc(x.ram_total_gb)} GB</div>
              </div>
              <div class="vast-kv">
                <div class="k">Ổ cứng (${esc(x.disk_name)})</div>
                <div class="v">${esc(x.disk_bw_mbs)} MB/s<br/>Disk ${esc(x.disk_used_gb)} / ${esc(x.disk_total_gb)} GB<br/>Port 7860 → ${x.port_7860 ? esc(x.port_7860) : "Đang mở..."}</div>
              </div>
            </div>

            <div class="vast-card-actions">
              <div style="display:flex; gap:8px; flex-wrap:wrap;">
                ${uiLink ? `<button type="button" class="btn ${isCurrent ? "secondary" : "primary"} sm" data-use-target="${esc(uiLink)}">${isCurrent ? "✓ Đang điều khiển máy này tại đây" : "⚡ Điều khiển ngay trên trang này"}</button>` : `<button type="button" class="btn ghost sm" disabled>⏳ Chờ container chạy...</button>`}
                ${uiLink ? `<a href="${esc(uiLink)}" target="_blank" class="btn ghost sm" style="text-decoration:none;" title="Mở tab riêng nếu cần">↗ Tab riêng</a>` : ""}
                <button type="button" class="btn secondary sm" data-logs="${x.id}">📜 Xem Log / Tunnel</button>
              </div>
              <button type="button" class="btn danger sm" data-destroy="${x.id}">🗑 Hủy máy</button>
            </div>

            <div class="vast-status-line">Status: ${esc(stMsg)}</div>
            <pre id="vastLog_${x.id}" class="logbox" style="display:${savedLog ? "block" : "none"}; max-height:180px;">${esc(savedLog)}</pre>
          </div>`;
      }).join("");

      wrap.querySelectorAll("[data-use-target]").forEach((btn) => {
        btn.onclick = () => setRemoteTarget(btn.dataset.useTarget, "add");
      });
      wrap.querySelectorAll("[data-destroy]").forEach((btn) => {
        btn.onclick = () => destroyVastInstance(Number(btn.dataset.destroy));
      });
      wrap.querySelectorAll("[data-logs]").forEach((btn) => {
        btn.onclick = () => loadVastLogs(Number(btn.dataset.logs));
      });
    } catch (err) {
      wrap.innerHTML = `<div class="muted" style="color:#fca5a5;">${esc(err.message || err)}</div>`;
    }
  }

  function ensureVastAutoPoll() {
    if (vastPollTimer) clearInterval(vastPollTimer);
    vastPollTimer = setInterval(() => {
      if (activeTab === "vast") refreshVastInstances();
    }, 4000);
  }
  ensureVastAutoPoll();

  async function destroyVastInstance(instanceId) {
    if (!confirm(`Hủy vĩnh viễn máy Vast.ai #${instanceId} để ngừng tính tiền?`)) return;
    try {
      const res = await fetch(`/api/vast/instances/${instanceId}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await res.text());
      await refreshVastInstances();
    } catch (err) {
      alert(`Hủy máy thất bại: ${err.message || err}`);
    }
  }

  // Tabs
  $("tabWork").onclick = () => switchTab("work");
  $("tabAdd").onclick = () => { switchTab("add"); loadDriveProjects(); };
  $("tabList").onclick = () => switchTab("list");
  $("tabVast").onclick = () => switchTab("vast");
  $("targetMachineSelect").onchange = () => setRemoteTarget($("targetMachineSelect").value);
  $("btnEmptyAdd").onclick = () => { switchTab("add"); loadDriveProjects(); };
  $("btnListAdd").onclick = () => { switchTab("add"); loadDriveProjects(); };
  $("btnRefreshDrive").onclick = loadDriveProjects;
  $("btnPullDrive").onclick = pullDriveJob;
  $("btnSearchVast").onclick = searchVastOffers;
  $("btnRefreshInstances").onclick = refreshVastInstances;

  // Drag & Drop + Browse
  const dz = $("dropzone");
  const fi = $("fileInput");
  $("btnBrowse").onclick = (e) => { e.stopPropagation(); fi.click(); };
  dz.onclick = () => fi.click();
  fi.onchange = () => { if (fi.files?.[0]) setPickedFile(fi.files[0]); };
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = (e) => {
    e.preventDefault();
    dz.classList.remove("over");
    if (e.dataTransfer.files?.[0]) setPickedFile(e.dataTransfer.files[0]);
  };
  $("downloadSelect").onchange = () => { if ($("downloadSelect").value) setPickedFile(null); };

  $("btnStartTest").onclick = () => submitNewJob("test_5min");
  $("btnStartFull").onclick = () => submitNewJob("full");

  $("btnOpenFolder").onclick = () => postJobAction("open-folder");
  $("actTest5").onclick = () => postJobAction("start", { mode: "test_5min", force_step1: false, target_stage: "auto" });
  $("actTestS3").onclick = () => postJobAction("start", { mode: "test_5min", force_step1: false, target_stage: "s3" });
  $("actFullAll").onclick = () => postJobAction("start", { mode: "full", force_step1: false, target_stage: "all" });
  $("actForceS1").onclick = () => {
    const mode = confirm("OK = Chạy lại FULL Bước 1\nCancel = Chạy lại TEST 5 PHÚT") ? "full" : "test_5min";
    postJobAction("start", { mode, force_step1: true, target_stage: "s1" });
  };
  $("actPushDrive").onclick = () => postJobAction("push-drive");
  $("actPushOutput").onclick = () => postJobAction("push-output");

  loadDownloads();
  refreshVastInstances();
  if (remoteTarget) setRemoteTarget(remoteTarget);
  connectWS();
})();
