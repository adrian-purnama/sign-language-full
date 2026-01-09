import React, { useState, useRef, useEffect } from 'react';
import { FaComments, FaPaperPlane } from 'react-icons/fa';
import '../styles/animations.css';

export interface ChatMessage {
  id: string;
  text: string;
  timestamp: number;
  sender: 'user' | 'disabled';
}

interface ChatPanelProps {
  isDisabled: boolean;
  onSendMessage: (message: string) => void;
  messages: ChatMessage[];
}

export const ChatPanel: React.FC<ChatPanelProps> = React.memo(({
  isDisabled,
  onSendMessage,
  messages,
}) => {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !isDisabled) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (isDisabled) {
    return (
      <div className="chat-panel">
        <div className="chat-header">
          <h3>
            <FaComments className="header-icon" />
            Messages
          </h3>
        </div>
        <div className="chat-content">
          {messages.length === 0 ? (
            <div className="chat-empty">
              <FaComments className="empty-icon" />
              <p>No messages yet</p>
              <span>Messages from others will appear here</span>
            </div>
          ) : (
            <div className="chat-messages">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`chat-message ${message.sender === 'user' ? 'message-received' : ''}`}
                >
                  <div className="message-content">
                    <div className="message-text">{message.text}</div>
                    <div className="message-time">{formatTime(message.timestamp)}</div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <h3>
          <FaComments className="header-icon" />
          Send Message
        </h3>
      </div>
      <div className="chat-content">
        {messages.length > 0 && (
          <div className="chat-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`chat-message ${message.sender === 'user' ? 'message-sent' : 'message-received'}`}
              >
                <div className="message-content">
                  <div className="message-text">{message.text}</div>
                  <div className="message-time">{formatTime(message.timestamp)}</div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
        <form onSubmit={handleSubmit} className="chat-input-form">
          <div className="chat-input-container">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type a message to send..."
              className="chat-input"
              maxLength={500}
            />
            <button
              type="submit"
              disabled={!inputValue.trim()}
              className="chat-send-button"
              title="Send message"
            >
              <FaPaperPlane />
            </button>
          </div>
          <div className="chat-hint">
            Your messages will be displayed to disabled users
          </div>
        </form>
      </div>
    </div>
  );
});

