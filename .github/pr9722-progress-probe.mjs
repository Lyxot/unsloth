import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source = readFileSync(new URL('../studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts', import.meta.url), 'utf8');
let poll = source.slice(source.indexOf('const pollDownload = async () =>'), source.indexOf('const pollLoad = async () =>'));
if (process.env.MUTATE_COMPLETION) poll = poll.replace('progressResponses.every(', 'progressResponses.some(');
const make = new Function('getDownloadProgress', 'progressModelIds', `
 const modelId = 'adapter', ggufVariant = null, expectedBytes = 0, hfToken = 'test-token';
 const abortCtrl = new AbortController(), loadingModelRef = {current: true};
 const loadToastDismissedRef = {current: true};
 let progressInterval = null, hasShownProgress = false, downloadComplete = false;
 let last;
 const setLoadProgress = p => last = p;
 const composeProgressLabel = () => 'progress';
 const dlSamples = [];
 const estimate = () => ({stable: false});
 ${poll}
 return {poll: pollDownload, state: () => ({downloadComplete, last}), targets: ids => progressModelIds = ids};
`);
const progress = n => ({progress:n,downloaded_bytes:n*100,expected_bytes:100});
for (const values of [[1,0.3],[0.3,1],[0,1],[1,0],[1,1]]) {
 const run = make(async id => progress(values[id === 'adapter' ? 0 : 1]), ['adapter','base']);
 await run.poll();
 assert.equal(run.state().downloadComplete, values.every(v => v === 1), JSON.stringify(values));
}
const cached = make(async () => progress(1), ['base']);
await cached.poll();
assert.equal(cached.state().last.phase, 'starting');
let resolve;
const stale = make(() => new Promise(r => resolve = r), ['adapter']);
const pending = stale.poll();
stale.targets(['base']);
resolve(progress(1));
await pending;
assert.equal(stale.state().downloadComplete, false);
assert.equal(stale.state().last, undefined);
console.log('PASS: 7 executed polling scenarios, including both adapter/base directions and stale response');
