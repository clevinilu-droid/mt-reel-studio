const $ = (id) => document.getElementById(id);
let currentPreset = "balanced";
let selectedOffer = null;
let lastOffers = [];

function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 3500);
}

// Tab switching
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    $(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "pipeline" || btn.dataset.tab === "storage") loadProjects();
    if (btn.dataset.tab === "docs") loadDocs();
  });
});

// Presets
document.querySelectorAll(".btn.preset").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".btn.preset").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentPreset = btn.dataset.preset;
    if (currentPreset === "fast") {
      $("fVram").value = "16";
      $("fMaxDph").value = "0.28";
      $("fMinDown").value = "400";
    } else if (currentPreset === "asia") {
      $("fVram").value = "12";
      $("fMaxDph").value = "0.25";
      $("fMinDown").value = "200";
    } else {
      $("fVram").value = "12";
      $("fMaxDph").value = "0.22";
      $("fMinDown").value = "200";
    }
    loadOffers();
  });
});

async function loadVastStatus() {
  try {
    const res = await fetch("/api/vast/status");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Lỗi Vast.ai");

    $("creditPill").innerHTML = `💳 Số dư: <b>$${data.credit !== null ? data.credit.toFixed(2) : "--"}</b>`;
    const n = (data.instances || []).length;
    $("runningPill").innerHTML = `🖥️ Đang thuê: <b>${n}</b>`;
    $("runningPill").classList.toggle("alert", n > 0);

    const box = $("instancesList");
    if (!n) {
      box.innerHTML = `<div class="empty-box">✅ Không có máy GPU nào đang chạy (Chi phí hiện tại: <b>$0.00/giờ</b>).</div>`;
      return;
    }

    box.innerHTML = data.instances.map((inst) => {
      const id = inst.ID || inst.Id || "--";
      const model = inst.Model || "GPU";
      const status = inst.Status || inst.State || "--";
      const dph = inst["$/hr"] || "--";
      const sshAddr = inst["SSH Addr"] || "";
      const sshPort = inst["SSH Port"] || "";
      const sshCmd = sshAddr && sshPort ? `ssh -p ${sshPort} root@${sshAddr}` : "";
      return `
        <div class="inst-card">
          <div class="inst-top">
            <div><b>#${id} · ${model}</b> <span class="badge amber">${status}</span></div>
            <div class="price-main">$${dph}/hr</div>
          </div>
          <div class="muted">${sshCmd ? `<code>${sshCmd}</code>` : "Đang cấp cổng SSH..."}</div>
          <div class="inst-actions">
            ${sshCmd ? `<button class="btn sm ghost" onclick="navigator.clipboard.writeText('${sshCmd}'); toast('Đã copy lệnh SSH!');">📋 Copy SSH</button>` : ""}
            <button class="btn sm danger" onclick="destroyInst(${id})">🗑️ Destroy Ngừng Tính Tiền</button>
          </div>
        </div>
      `;
    }).join("");
  } catch (e) {
    $("instancesList").innerHTML = `<div class="empty-box">⚠️ ${e.message}</div>`;
  }
}

async function destroyInst(id) {
  if (!confirm(`Xác nhận hủy vĩnh viễn máy #${id} để dừng tính tiền?`)) return;
  try {
    const res = await fetch(`/api/vast/destroy/${id}`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Lỗi hủy máy");
    toast(`Đã hủy máy #${id}!`);
    loadVastStatus();
  } catch (e) {
    toast("Lỗi: " + e.message);
  }
}

async function loadOffers() {
  const tbody = $("offersTbody");
  tbody.innerHTML = `<tr><td colspan="9" class="center muted">⏳ Đang quét bảng giá GPU live từ Vast.ai...</td></tr>`;
  const params = new URLSearchParams({
    preset: currentPreset,
    min_vram: $("fVram").value,
    max_dph: $("fMaxDph").value,
    min_disk: $("fMinDisk").value,
    min_down: $("fMinDown").value,
    exclude_cn: $("fExcludeCn").checked ? "true" : "false",
    limit: "15",
  });
  try {
    const res = await fetch(`/api/vast/offers?${params}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Không tải được GPU");
    $("queryStrBox").textContent = `${data.query} (-o ${data.sort})`;

    lastOffers = data.offers || [];
    if (!lastOffers.length) {
      tbody.innerHTML = `<tr><td colspan="9" class="center muted">Không có máy nào khớp bộ lọc — thử tăng Giá tối đa hoặc giảm Mạng.</td></tr>`;
      return;
    }

    tbody.innerHTML = lastOffers.map((o, idx) => {
      const id = o.ID;
      const model = o.Model || "";
      const vram = o.VRAM || "?";
      const dph = parseFloat(o["$/hr"] || "0");
      const vnd = Math.round(dph * 25500).toLocaleString("vi-VN");
      const dlp = parseFloat(o.DLP || "0");
      const dlpUsd = parseFloat(o["DLP/$"] || "0");
      const pct = Math.min(100, Math.round((dlpUsd / 450) * 100));
      const vcpu = o.vCPUs || "?";
      const ram = o.RAM || "?";
      const disk = o.Disk || "?";
      const netUp = o.Net_up || "?";
      const netDown = o.Net_down || "?";
      const country = (o.country || "").replaceAll("_", " ");
      const rel = o.R || "?";

      let genBadge = `<span class="badge blue">Ampere</span>`;
      if (/\b50\d{2}\b/.test(model)) genBadge = `<span class="badge green">Blackwell 50xx</span>`;
      else if (/\b40\d{2}\b/.test(model) || /\bL40?S?\b/i.test(model)) genBadge = `<span class="badge blue">Ada 40xx</span>`;

      const isTop = dlpUsd >= 350 || idx === 0;
      return `
        <tr class="${isTop ? "top-pick" : ""}">
          <td><b>${idx + 1}</b></td>
          <td><code>${id}</code></td>
          <td><b>${model}</b> ${genBadge}<div class="muted">${vram} GB VRAM · CUDA ${o.CUDA || "?"}</div></td>
          <td><div class="price-main">$${dph.toFixed(4)}</div><div class="price-vnd">~${vnd} đ/giờ</div></td>
          <td>
            <b>${dlpUsd.toFixed(1)}</b> <span class="muted">(DLP ${dlp})</span>
            <div class="dlp-bar"><div class="dlp-fill" style="width:${pct}%"></div></div>
          </td>
          <td>${vcpu} vCPU · ${ram} GB<div class="muted">Disk: ${disk} GB · PCIe ${o.PCIE || "?"}</div></td>
          <td>⬆ ${netUp} <div class="muted">⬇ ${netDown} Mbps</div></td>
          <td><b>${country}</b><div class="muted">Tin cậy: ${rel}%</div></td>
          <td><button class="btn primary sm" onclick="openRentModal(${idx})">🚀 Thuê</button></td>
        </tr>
      `;
    }).join("");
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="9" class="center muted">⚠️ ${e.message}</td></tr>`;
  }
}

function openRentModal(idx) {
  const offer = lastOffers[idx];
  if (!offer) return;
  selectedOffer = offer;
  const dph = parseFloat(offer["$/hr"] || "0");
  $("rentDiskInput").value = Math.max(60, parseInt($("fMinDisk").value || "60", 10));
  $("rentModalBody").innerHTML = `
    <div><b>Máy #${offer.ID}:</b> ${offer.Model} (${offer.VRAM} GB VRAM) — <b>${(offer.country || "").replaceAll("_", " ")}</b></div>
    <div><b>Giá thuê:</b> $${dph.toFixed(4)}/giờ · <b>Mạng:</b> ${offer.Net_up}/${offer.Net_down} Mbps</div>
    <div style="margin-top:8px;"><b>Image sẽ nạp sẵn:</b> <code>ghcr.io/clevinilu-droid/mt-reel-vast:latest</code></div>
  `;
  $("rentModal").style.display = "grid";
}

$("btnCancelRent").addEventListener("click", () => ($("rentModal").style.display = "none"));
$("btnConfirmRent").addEventListener("click", async () => {
  if (!selectedOffer) return;
  const btn = $("btnConfirmRent");
  btn.disabled = true;
  btn.textContent = "⏳ Đang khởi tạo...";
  try {
    const res = await fetch("/api/vast/rent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        offer_id: parseInt(selectedOffer.ID, 10),
        disk_gb: parseInt($("rentDiskInput").value, 10),
        label: $("rentLabelInput").value || "mt-reel-step3",
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Thuê máy thất bại");
    $("rentModal").style.display = "none";
    toast("✅ Đã gửi lệnh thuê máy thành công!");
    loadVastStatus();
  } catch (e) {
    toast("Lỗi thuê máy: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "✅ Bấm Thuê Ngay";
  }
});

async function loadProjects() {
  try {
    const res = await fetch("/api/projects");
    const data = await res.json();

    // Tab 3 Downloads
    $("downloadsList").innerHTML = (data.downloads || []).length
      ? data.downloads.map((d) => `<div class="list-row"><span>🎬 ${d.rel_path}</span><b>${d.size_mb} MB</b></div>`).join("")
      : `<div class="empty-box">Chưa có file .mp4 trong Download/</div>`;

    // Tab 2 Pipeline
    const cont = $("projectsContainer");
    if (!(data.projects || []).length) {
      cont.innerHTML = `<div class="empty-box">Chưa có dự án nào trong projects/</div>`;
      return;
    }

    cont.innerHTML = data.projects.map((proj) => {
      const partsHtml = proj.parts.map((p) => {
        const s3Done = p.has_zh_synced && p.has_vocals && p.has_instr;
        const s3Part = p.has_zh_raw || p.has_test_10m;
        return `
          <div style="margin-top:14px;">
            <div class="section-head" style="margin-bottom:8px;">
              <div><b>Phần ${p.part}</b></div>
              ${p.has_test_10m ? `<span class="badge blue">Có bản test 10 phút</span>` : ""}
            </div>
            <div class="steps-grid">
              <div class="step-box ${p.step1_norm ? "done" : ""}">
                <div class="step-title">Bước 1 · Chuẩn Hóa Nguồn ${p.step1_norm ? "✅" : "⏳"}</div>
                <div class="step-items">
                  <span>• Master 1080×1920 30fps: ${p.step1_norm ? `✅ (${p.norm_mb} MB)` : "❌ Chưa chuẩn hóa"}</span>
                </div>
              </div>
              <div class="step-box ${p.step2_done ? "done" : ""}">
                <div class="step-title">Bước 2 · Nhân vật & Blur ${p.step2_done ? "✅" : "⏳"}</div>
                <div class="step-items">
                  <span>• Hồ sơ nhân vật: ${proj.has_profile ? "✅" : "❌"}</span>
                  <span>• Tọa độ blur (Softsub): ${p.step2_done ? "✅" : "❌"}</span>
                </div>
              </div>
              <div class="step-box ${s3Done ? "done" : s3Part ? "partial" : ""}">
                <div class="step-title">Bước 3 · Hardsub & Tách âm ${s3Done ? "✅" : s3Part ? "🟡 Đang dở" : "⏳"}</div>
                <div class="step-items">
                  <span>• Hardsub raw (.srt): ${p.has_zh_raw ? "✅ Đã có (180 câu 10m)" : "❌ Chưa tách"}</span>
                  <span>• Tách giọng RoFormer (.flac): ${p.has_vocals ? "✅" : "❌ Chờ Vast.ai"}</span>
                  <span>• Tách BGM (.flac): ${p.has_instr ? "✅" : "❌ Chờ Vast.ai"}</span>
                  <span>• Khớp sóng âm (Bước 3.4): ${p.has_zh_synced ? "✅" : "❌ Chờ vocals"}</span>
                </div>
              </div>
              <div class="step-box ${p.has_vi_srt ? "done" : ""}">
                <div class="step-title">Bước 4 · Dịch Đa Ngôn Ngữ ${p.has_vi_srt ? "✅" : "⏳"}</div>
                <div class="step-items">
                  <span>• Bản dịch Việt (Chốt #2): ${p.has_vi_srt ? "✅" : "❌ Chưa dịch"}</span>
                </div>
              </div>
              <div class="step-box ${p.outputs.length ? "done" : ""}">
                <div class="step-title">Bước 5 · VieNeu & Xuất YouTube ${p.outputs.length ? "✅" : "⏳"}</div>
                <div class="step-items">
                  <span>• Thành phẩm: ${p.outputs.length ? p.outputs.join(", ") : "Chưa xuất"}</span>
                </div>
              </div>
            </div>
            ${p.blur_preview_url ? `
              <details style="margin-top:12px;">
                <summary style="cursor:pointer; color:#93c5fd;">🖼️ Xem lại ảnh mẫu làm mờ đáy đã duyệt (Bước 2)</summary>
                <img class="preview-img" src="${p.blur_preview_url}" alt="Blur preview" />
              </details>` : ""}
          </div>
        `;
      }).join("");

      return `
        <div class="card">
          <div class="section-head">
            <div>
              <h2>🎬 ${proj.series}</h2>
              <p class="muted">Từ điển nhân vật: ${proj.has_glossary ? "✅ Đã khóa character_glossary.json" : "❌ Chưa có"}</p>
            </div>
          </div>
          ${partsHtml}
        </div>
      `;
    }).join("");
  } catch (e) {
    $("projectsContainer").innerHTML = `<div class="empty-box">⚠️ ${e.message}</div>`;
  }
}

async function loadDrive() {
  $("driveList").innerHTML = `<div class="empty-box">⏳ Đang quét Google Drive...</div>`;
  try {
    const res = await fetch("/api/drive/list");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Lỗi Drive");
    $("driveList").innerHTML = data.items.length
      ? data.items.map((it) => `<div class="list-row"><span>☁️ ${it.path}</span><b>${it.size_mb} MB</b></div>`).join("")
      : `<div class="empty-box">Thư mục gdrive:MT-Video đang trống</div>`;
  } catch (e) {
    $("driveList").innerHTML = `<div class="empty-box">⚠️ ${e.message}</div>`;
  }
}

async function loadDocs() {
  const res = await fetch("/api/workflow-doc");
  const data = await res.json();
  $("docContent").textContent = data.content || "Chưa có tài liệu.";
}

$("btnRefreshStatus").addEventListener("click", loadVastStatus);
$("btnSearchOffers").addEventListener("click", loadOffers);
$("btnReloadProjects").addEventListener("click", loadProjects);
$("btnReloadDrive").addEventListener("click", loadDrive);

loadVastStatus();
loadOffers();
loadProjects();
