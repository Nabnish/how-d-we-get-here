import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    setMessages(prev => [...prev, { role: 'user', content: input }])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      })

      const data = await response.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Backend error. Check Flask server.' }
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setIsLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('http://localhost:5000/api/upload-pdf', {
        method: 'POST',
        body: formData
      })

      const data = await res.json()

      const explainRes = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `Explain this medical report in simple language:\n\n${data.text}`
        })
      })

      const explainData = await explainRes.json()

      setMessages(prev => [
        ...prev,
        { role: 'user', content: `Uploaded: ${file.name}` },
        { role: 'assistant', content: explainData.response }
      ])
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Error processing PDF.' }
      ])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">🤖 Medical Records LLM</h1>
      </header>

      <main className="chat-container">
        <div className="messages-list">
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-avatar">{msg.role === 'user' ? '👤' : '🤖'}</div>
              <div className="message-text">{msg.content}</div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant">
              <div className="message-avatar">🤖</div>
              <div className="message-text">Thinking…</div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="input-container">
        <form onSubmit={handleSend} className="input-form">

          {/* PDF Upload */}
          <label className="upload-pill">
            📄
            <input
              type="file"
              accept=".pdf"
              hidden
              disabled={isLoading}
              onChange={handleFileUpload}
            />
          </label>

          {/* Chat Input */}
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask or upload a medical report…"
            className="message-input"
            disabled={isLoading}
          />

          {/* Send */}
          <button
            type="submit"
            className="send-button"
            disabled={isLoading || !input.trim()}
          >
            ➤
          </button>

        </form>
      </footer>
    </div>
  )
}

export default App
