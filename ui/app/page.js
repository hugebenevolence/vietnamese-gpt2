'use client';

import { useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export default function HomePage() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function sendMessage(event) {
    event.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', text: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.text })
      });

      if (!response.ok) {
        throw new Error(`Request failed with ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [...prev, { role: 'assistant', text: data.reply }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: `Lỗi gọi API: ${error.message}` }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>Vietnamese GPT-2 Chat</h1>
      <div className="chat-box">
        {messages.length === 0 ? (
          <p className="placeholder">Nhập câu hỏi để bắt đầu chat.</p>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <strong>{msg.role === 'user' ? 'Bạn' : 'Bot'}:</strong> {msg.text}
            </div>
          ))
        )}
      </div>
      <form onSubmit={sendMessage} className="chat-form">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Nhập nội dung..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Đang gửi...' : 'Gửi'}
        </button>
      </form>
    </main>
  );
}
