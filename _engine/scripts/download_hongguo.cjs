/**
 * Headless CLI downloader for Hongguo Drama using jackotom/hongguo-drama native engine.
 * Usage:
 *   node scripts/download_hongguo.cjs <series_id> <start_ep> <end_ep> <out_dir> [quality]
 */
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { pipeline } = require('node:stream/promises');

const ENGINE_DIR = path.resolve(__dirname, '..', 'tools', 'hongguo-engine');
const axios = require(path.join(ENGINE_DIR, 'node_modules', 'axios'));
const hongguo = require(path.join(ENGINE_DIR, 'src', 'native', 'hongguo'));

function pad3(n) {
  return String(n).padStart(3, '0');
}

async function downloadEpisode(seriesId, ep, outFile, quality = '1080p') {
  if (fs.existsSync(outFile) && fs.statSync(outFile).size > 500000) {
    console.log(`[SKIP] Episode ${ep.vid_index} already exists: ${path.basename(outFile)}`);
    return true;
  }

  let info = null;
  let lastErr = null;
  for (let attempt = 1; attempt <= 4; attempt++) {
    try {
      console.log(`[INFO] Fetching stream URL for Ep ${ep.vid_index} (vid=${ep.vid}, attempt ${attempt}/4)...`);
      info = await hongguo.fetchPlayUrlSingle(String(ep.vid), String(seriesId), { quality });
      if (info && info.url && !info.locked) break;
      lastErr = new Error(info?.error || 'no url / locked');
    } catch (e) {
      lastErr = e;
    }
    if (attempt < 4) {
      const waitMs = attempt * 3500;
      console.log(`[WARN] Attempt ${attempt} failed (${lastErr?.message}), waiting ${waitMs}ms...`);
      await new Promise(r => setTimeout(r, waitMs));
    }
  }
  if (!info || !info.url || info.locked) {
    throw new Error(`Failed to fetch Ep ${ep.vid_index}: ${lastErr?.message || 'no url / locked'}`);
  }
  console.log(`[INFO] Ep ${ep.vid_index}: codec=${info.codec || 'unknown'}, ${info.width}x${info.height}, duration=${info.duration}s`);

  const tmpFile = outFile + '.enc.tmp';
  const outTmp = outFile + '.dec.tmp';

  try {
    const resp = await axios.get(info.url, {
      responseType: 'stream',
      timeout: 120000,
      headers: {
        'User-Agent': info.userAgent || hongguo.UA,
        'Accept-Encoding': 'identity',
        ...(info.referer ? { Referer: info.referer } : {}),
      },
    });
    if (resp.status !== 200) {
      throw new Error(`HTTP ${resp.status} while downloading Ep ${ep.vid_index}`);
    }
    await pipeline(resp.data, fs.createWriteStream(tmpFile));

    const key = info.contentKey || (info.spadeA ? hongguo.deriveKey(info.spadeA)?.toString('hex') : null);
    if (key) {
      console.log(`[INFO] Decrypting Ep ${ep.vid_index} with ffmpeg...`);
      execFileSync(
        'ffmpeg',
        [
          '-y', '-v', 'error', '-xerror', '-nostdin',
          '-decryption_key', key,
          '-i', tmpFile,
          '-map', '0:v:0', '-map', '0:a?',
          '-c', 'copy', '-movflags', '+faststart',
          '-f', 'mp4', outTmp,
        ],
        { stdio: 'inherit', timeout: 120000 }
      );
      fs.renameSync(outTmp, outFile);
    } else {
      fs.renameSync(tmpFile, outFile);
    }

    const sizeMB = (fs.statSync(outFile).size / 1024 / 1024).toFixed(2);
    console.log(`[OK] Saved Ep ${ep.vid_index} (${sizeMB} MB) -> ${outFile}`);
    return true;
  } finally {
    if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
    if (fs.existsSync(outTmp)) fs.unlinkSync(outTmp);
  }
}

(async () => {
  const [seriesId, startStr, endStr, outDir, quality = '1080p'] = process.argv.slice(2);
  if (!seriesId || !startStr || !endStr || !outDir) {
    console.error('Usage: node download_hongguo.cjs <series_id> <start_ep> <end_ep> <out_dir> [quality]');
    process.exit(1);
  }
  const startEp = parseInt(startStr, 10);
  const endEp = parseInt(endStr, 10);
  fs.mkdirSync(outDir, { recursive: true });

  console.log(`[INFO] Fetching episode list for series ${seriesId}...`);
  const detail = await hongguo.fetchEpisodeList(String(seriesId), { deadline: Date.now() + 30000 });
  const title = detail.series_title || `series_${seriesId}`;
  console.log(`[INFO] Series: ${title} | Total episodes: ${detail.episodes.length}`);

  const targets = detail.episodes.filter(ep => ep.vid_index >= startEp && ep.vid_index <= endEp);
  if (!targets.length) {
    throw new Error(`No episodes found in range ${startEp}-${endEp}`);
  }

  for (const ep of targets) {
    const safeTitle = title.replace(/[\\/:*?"<>|]/g, '_');
    const outFile = path.join(outDir, `${safeTitle} ${pad3(ep.vid_index)}.mp4`);
    const existed = fs.existsSync(outFile) && fs.statSync(outFile).size > 500000;
    await downloadEpisode(seriesId, ep, outFile, quality);
    if (!existed) {
      await new Promise(r => setTimeout(r, 2500));
    }
  }
  console.log('[DONE] All requested episodes downloaded!');
})().catch(err => {
  console.error('[ERROR]', err?.message || err);
  process.exit(1);
});
