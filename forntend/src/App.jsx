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

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please make sure the backend server is running.' 
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = () => {
    setMessages([])
  }
  

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">
            <span className="title-icon">🤖</span>
            Medical Records LLM Assistant
          </h1>
          <p className="app-subtitle">Ask questions about medical research and get intelligent answers</p>
        </div>
        {messages.length > 0 && (
          <button className="clear-button" onClick={handleClear}>
            Clear Chat
          </button>
        )}
      </header>

      <main className="chat-container">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-content">
              <div className="welcome-icon">💬</div>
              <h2>Welcome to Medical Records LLM</h2>
              <p>Start a conversation by asking a question about medical research, treatments, or any health-related topic.</p>
              <div className="example-questions">
                <p className="examples-title">Try asking:</p>
                <div className="example-chips">
                  <button 
                    className="example-chip"
                    onClick={() => setInput("What are the latest treatments for diabetes?")}
                  >
                    Latest diabetes treatments
                  </button>
                  <button 
                    className="example-chip"
                    onClick={() => setInput("Explain the symptoms of COVID-19")}
                  >
                    COVID-19 symptoms
                  </button>
                  <button 
                    className="example-chip"
                    onClick={() => setInput("What is immunotherapy?")}
                  >
                    What is immunotherapy?
                  </button>
                  <button 
                    className="example-chip"
                    onClick={() => setInput("Recent advances in cancer research")}
                  >
                    Cancer research advances
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-text">{msg.content}</div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      <footer className="input-container">
        <form onSubmit={handleSend} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about medical research..."
            className="message-input"
            disabled={isLoading}
          />
          <button 
            type="submit" 
            className="send-button"
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? '⏳' : '➤'}
          </button>
        </form>
      </footer>
    </div>
  )
}

export default App
