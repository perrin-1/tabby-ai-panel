import { test } from 'node:test'
import assert from 'node:assert/strict'
import * as http from 'node:http'
import { OpenAICompatibleProvider, SESSION_ID_HEADER } from '../src/core/llm/openai-compatible'
import { createSession } from '../src/core/session-utils'

interface Seen {
    method: string
    url: string
    headers: http.IncomingHttpHeaders
    body: string
}

/** Fake server recording every request; `handler` writes the response. */
async function withServer (
    handler: (req: http.IncomingMessage, res: http.ServerResponse, body: string) => void,
    fn: (endpoint: string, seen: Seen[]) => Promise<void>,
): Promise<void> {
    const seen: Seen[] = []
    const server = http.createServer((req, res) => {
        let body = ''
        req.on('data', c => { body += c })
        req.on('end', () => {
            seen.push({ method: req.method ?? '', url: req.url ?? '', headers: req.headers, body })
            handler(req, res, body)
        })
    })
    await new Promise<void>(r => server.listen(0, r))
    const port = (server.address() as any).port
    try {
        await fn(`http://127.0.0.1:${port}`, seen)
    } finally {
        server.closeAllConnections?.()
        server.close()
    }
}

function sse (res: http.ServerResponse): void {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' })
    res.write('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n')
    res.write('data: [DONE]\n\n')
    res.end()
}

async function drain (provider: OpenAICompatibleProvider): Promise<void> {
    for await (const _ of provider.stream([{ role: 'user', content: 'hi' }], [])) { /* drain */ }
}

test('chat completions carry the session id header, not in the body', async () => {
    const sessionId = createSession().id
    await withServer((_req, res) => sse(res), async (endpoint, seen) => {
        const provider = new OpenAICompatibleProvider({ endpoint, apiKey: 'k', model: 'm', sessionId })
        await drain(provider)
        assert.equal(seen.length, 1)
        assert.equal(seen[0].headers[SESSION_ID_HEADER], sessionId)
        assert.equal(seen[0].headers.authorization, 'Bearer k')
        assert.equal(seen[0].headers['content-type'], 'application/json')
        assert.ok(!seen[0].body.includes(sessionId), 'session id must not leak into the request body')
    })
})

test('every request of one provider (tool rounds) reuses the same session id', async () => {
    await withServer((_req, res) => sse(res), async (endpoint, seen) => {
        const provider = new OpenAICompatibleProvider({ endpoint, model: 'm', sessionId: 's_abc123' })
        await drain(provider)
        await drain(provider)
        await drain(provider)
        assert.deepEqual(seen.map(s => s.headers[SESSION_ID_HEADER]), ['s_abc123', 's_abc123', 's_abc123'])
    })
})

test('model list requests carry the session id header', async () => {
    await withServer((_req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ data: [{ id: 'a' }] }))
    }, async (endpoint, seen) => {
        const provider = new OpenAICompatibleProvider({ endpoint, model: 'm', sessionId: 's_models' })
        const models = await provider.listModels()
        assert.deepEqual(models?.map(m => m.id), ['a'])
        assert.equal(seen[0].method, 'GET')
        assert.equal(seen[0].url, '/v1/models')
        assert.equal(seen[0].headers[SESSION_ID_HEADER], 's_models')
    })
})

test('connection test fallback completion carries the session id header', async () => {
    await withServer((req, res) => {
        if (req.url === '/v1/models') {
            res.writeHead(404)
            res.end()
            return
        }
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(JSON.stringify({ choices: [{ message: { content: 'p' } }] }))
    }, async (endpoint, seen) => {
        const provider = new OpenAICompatibleProvider({ endpoint, model: 'm', sessionId: 's_ping' })
        assert.deepEqual(await provider.ping(), { ok: true })
        assert.deepEqual(seen.map(s => s.url), ['/v1/models', '/v1/chat/completions'])
        assert.deepEqual(seen.map(s => s.headers[SESSION_ID_HEADER]), ['s_ping', 's_ping'])
    })
})

test('the header survives a redirect', async () => {
    await withServer((req, res) => {
        if (req.url === '/v1/chat/completions') {
            res.writeHead(307, { Location: '/moved/v1/chat/completions' })
            res.end()
            return
        }
        sse(res)
    }, async (endpoint, seen) => {
        const provider = new OpenAICompatibleProvider({ endpoint, model: 'm', sessionId: 's_redirect' })
        await drain(provider)
        assert.deepEqual(seen.map(s => s.url), ['/v1/chat/completions', '/moved/v1/chat/completions'])
        assert.deepEqual(seen.map(s => s.headers[SESSION_ID_HEADER]), ['s_redirect', 's_redirect'])
    })
})

test('no header without a usable session id', async () => {
    await withServer((_req, res) => sse(res), async (endpoint, seen) => {
        await drain(new OpenAICompatibleProvider({ endpoint, model: 'm' }))
        await drain(new OpenAICompatibleProvider({ endpoint, model: 'm', sessionId: '   ' }))
        await drain(new OpenAICompatibleProvider({ endpoint, model: 'm', sessionId: 'bad\nid' }))
        assert.equal(seen.length, 3)
        for (const s of seen) {
            assert.equal(s.headers[SESSION_ID_HEADER], undefined)
        }
    })
})

test('session ids are valid header values and differ per session', () => {
    const a = createSession().id
    const b = createSession().id
    assert.match(a, /^s_[0-9a-z]+$/)
    assert.notEqual(a, b)
})
