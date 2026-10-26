import { useEffect, useState } from 'react'

// 2.0 packages listed under `voice_v2_submissions` in manifest.json, plus the
// separately scored banking and entity packages listed under
// `voice_v2_domain_submissions`. Each directory holds the files
// `tau2 submit prepare` writes; see
// src/tau2/scripts/leaderboard/benchmark_package.py.
const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`
const NO_CACHE = { cache: 'no-cache' }
const FILES = {
  metadata: 'benchmark-metadata.json',
  submission: 'benchmark-submission.json',
  diagnostics: 'benchmark-diagnostics.json',
  quality: 'benchmark-quality.json',
  pass3: 'benchmark-pass3.json',
}
// Files a package may omit, requested only for the frames that can carry them
// (benchmark_package.py). A missing one loads as null; some static servers
// answer a missing file with the HTML index, which is treated the same way.
// Diagnostics are the largest file and only feed a breakdown, so they load on
// demand (loadDiagnostics), not with the package.
const OPTIONAL_BY_FRAME = {
  'voice-v2': ['quality'],
  'voice-v2-entity': ['pass3'],
}
const OPTIONAL = new Set(['diagnostics', ...Object.values(OPTIONAL_BY_FRAME).flat()])

export const LANGUAGES = [
  { id: 'en', name: 'English' }, { id: 'es', name: 'Spanish' }, { id: 'pt', name: 'Portuguese' },
  { id: 'hi', name: 'Hindi' }, { id: 'ko', name: 'Korean' }, { id: 'zh', name: 'Mandarin' },
]
export const CORE_DOMAINS = ['airline', 'retail', 'telecom']

// Domain benchmarks score domains outside the voice-v2 Overall. Their packages
// attach to the same agent's voice-v2 row; they never change its Overall.
export const DOMAIN_BENCHMARKS = { 'voice-v2-banking': 'banking_knowledge', 'voice-v2-entity': 'intake' }
// Domains scored by crossed Pass^3 (benchmark-pass3.json) instead of pass@1:
// a task passes only if it passes in all three voice conditions.
export const PASS3_DOMAINS = ['intake']

const providerLogo = { gemini: 'Google__G__logo.svg.png', openai: 'openai.svg', openai_live: 'openai.svg', xai: 'xai-logo.svg' }
export const toPercent = value => (value == null ? null : 100 * value)

async function loadFile(dir, key) {
  const file = FILES[key]
  const response = await fetch(`${SUBMISSIONS_BASE}/${dir}/${file}`, NO_CACHE)
  if (!response.ok) {
    if (OPTIONAL.has(key)) return [key, null]
    throw new Error(`${dir}/${file}: HTTP ${response.status}`)
  }
  if (!OPTIONAL.has(key)) return [key, await response.json()]
  try { return [key, await response.json()] } catch { return [key, null] }
}

// The required files first; the frame then names the optional files to request,
// so a package is never probed for a file its frame cannot carry.
async function loadPackage(dir, accepts) {
  const required = Object.keys(FILES).filter(key => !OPTIONAL.has(key))
  const pkg = Object.fromEntries(await Promise.all(required.map(key => loadFile(dir, key))))
  const { summary, frame } = pkg.submission
  if (!accepts(frame.benchmark_version) || !summary.complete) throw new Error(`${dir}: not a complete ${frame.benchmark_version} package of this list`)
  const optional = OPTIONAL_BY_FRAME[frame.benchmark_version] || []
  Object.assign(pkg, Object.fromEntries([...OPTIONAL].filter(key => key !== 'diagnostics').map(key => [key, null])))
  Object.assign(pkg, Object.fromEntries(await Promise.all(optional.map(key => loadFile(dir, key)))))
  return { dir, ...pkg }
}

// A voice-v2 package's benchmark-diagnostics.json, fetched once per package
// when a breakdown first needs it; resolves to null if the package has none.
const diagnosticsCache = new Map()
export function loadDiagnostics(dir) {
  if (!diagnosticsCache.has(dir)) {
    diagnosticsCache.set(dir, loadFile(dir, 'diagnostics').then(([, value]) => value, () => null))
  }
  return diagnosticsCache.get(dir)
}

// One submitted system: the same model, organization, agent and submission type.
const agentKey = ({ metadata }) => [metadata.model_name, metadata.model_organization, metadata.agent_provider, metadata.agent_model, metadata.submission_type || 'standard'].join('\u0000')

// Attach each domain package to the one voice-v2 package of the same agent.
// A domain package without one becomes its own row with no Overall. Rows are
// voice-v2-shaped: `submission` is the voice-v2 package (or null) and
// `extras` maps a domain to its separately scored package.
export function groupPackages(cores, domainPackages) {
  const errors = []
  // One row per agent: the first listed voice-v2 package wins.
  const byAgent = new Map()
  for (const pkg of cores) {
    const key = agentKey(pkg)
    if (byAgent.has(key)) errors.push(`${pkg.dir}: duplicate voice-v2 package for ${byAgent.get(key).dir}`)
    else byAgent.set(key, { ...pkg, extras: {} })
  }
  const rows = [...byAgent.values()]
  const standalone = new Map()
  for (const pkg of domainPackages) {
    const domain = DOMAIN_BENCHMARKS[pkg.submission.frame.benchmark_version]
    const key = agentKey(pkg)
    const core = byAgent.get(key)
    const row = core || standalone.get(key) || { dir: pkg.dir, metadata: pkg.metadata, submission: null, extras: {} }
    if (row.extras[domain]) {
      errors.push(`${pkg.dir}: duplicate ${domain} package for ${row.dir}`)
      continue
    }
    row.extras[domain] = pkg
    if (!core) standalone.set(key, row)
  }
  return { rows: [...rows, ...standalone.values()], errors }
}

// Loads every complete package in the manifest. `packages` is null while
// loading; `errors` names packages that were skipped.
export function useVoiceV2Packages() {
  const [packages, setPackages] = useState(null)
  const [errors, setErrors] = useState([])
  useEffect(() => {
    let cancelled = false
    const settle = results => [
      results.filter(result => result.status === 'fulfilled').map(result => result.value),
      results.filter(result => result.status === 'rejected').map(result => result.reason.message),
    ]
    const load = async () => {
      const response = await fetch(`${SUBMISSIONS_BASE}/manifest.json`, NO_CACHE)
      const manifest = await response.json()
      const [cores, coreErrors] = settle(await Promise.allSettled((manifest.voice_v2_submissions || []).map(dir => loadPackage(dir, version => version === 'voice-v2'))))
      const [extras, extraErrors] = settle(await Promise.allSettled((manifest.voice_v2_domain_submissions || []).map(dir => loadPackage(dir, version => version in DOMAIN_BENCHMARKS))))
      if (cancelled) return
      const grouped = groupPackages(cores, extras)
      setPackages(grouped.rows)
      setErrors([...coreErrors, ...extraErrors, ...grouped.errors])
    }
    load().catch(error => !cancelled && (setPackages([]), setErrors([error.message])))
    return () => { cancelled = true }
  }, [])
  return { packages, errors }
}

// Task-completion pass@1 (percent) for one language or 'all'; null if the
// package does not cover that language or has no voice-v2 Overall.
export const packageScore = (pkg, language) => pkg.submission && toPercent(language === 'all' ? pkg.submission.summary.score : pkg.submission.summary.by_language[language]?.score)

// Task completion (percent) for one domain and language or 'all', from the
// domain's own package when it has one, else from the voice-v2 package.
// Pass@1, except crossed Pass^3 for PASS3_DOMAINS; null if a Pass^3 domain's
// package carries no benchmark-pass3.json.
export function domainScore(pkg, domain, language = 'all') {
  if (PASS3_DOMAINS.includes(domain)) {
    const pass3 = pkg.extras?.[domain]?.pass3
    if (!pass3) return null
    return toPercent(language === 'all' ? pass3.score : pass3.cells.find(cell => cell.language === language && cell.domain === domain)?.score)
  }
  const summary = pkg.extras?.[domain]?.submission.summary || pkg.submission?.summary
  if (!summary) return null
  if (language === 'all') return toPercent(summary.by_domain[domain]?.score)
  return toPercent(summary.cells.find(cell => cell.language === language && cell.domain === domain)?.score)
}

// Interaction or Generation quality (`experience`) pass rate, in percent,
// pooled over the scored calls of the voice-v2 package's quality file:
// passed / (passed + failed). Unscored and excluded calls are left out. Scope
// is one language or 'all', and one core domain or 'all'; null if the package
// has no quality file or no scored call in scope.
const qualityCells = (pkg, language, domain) => (pkg.quality?.cells || [])
  .filter(cell => (language === 'all' || cell.language === language) && (domain === 'all' || cell.domain === domain))
export function packageQuality(pkg, measure, language = 'all', domain = 'all') {
  const cells = qualityCells(pkg, language, domain)
  const passed = cells.reduce((sum, cell) => sum + (cell[measure]?.passed || 0), 0)
  const failed = cells.reduce((sum, cell) => sum + (cell[measure]?.failed || 0), 0)
  return passed + failed ? 100 * passed / (passed + failed) : null
}

// Share of calls (percent) with one Interaction flag raised, over the calls
// where that flag is known.
export function interactionComponent(pkg, component, language = 'all', domain = 'all') {
  const cells = qualityCells(pkg, language, domain)
  const flagged = cells.reduce((sum, cell) => sum + (cell.interaction_components?.[component]?.flagged || 0), 0)
  const known = cells.reduce((sum, cell) => sum + (cell.interaction_components?.[component]?.known || 0), 0)
  return known ? 100 * flagged / known : null
}

// Calls counted toward a quality measure: scored, unscored and excluded.
export function qualityCounts(pkg, measure, language = 'all', domain = 'all') {
  const counts = { passed: 0, failed: 0, unscored: 0, excluded: 0 }
  for (const cell of qualityCells(pkg, language, domain)) for (const key of Object.keys(counts)) counts[key] += cell[measure]?.[key] || 0
  return counts
}

export const packageLogo = pkg => providerLogo[pkg.metadata.agent_provider]
