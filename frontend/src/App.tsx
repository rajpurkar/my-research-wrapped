import { useState } from 'react'
import './App.css'
import { YearInReview } from './components/YearInReview'
import { TopicsList } from './components/TopicsList'
import { PapersList } from './components/PapersList'

function App() {
  const [activeTab, setActiveTab] = useState<'year-review' | 'topics' | 'papers'>('year-review')

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>MyResearchWrapped</h1>
        <div className="header-controls">
          <div className="view-tabs">
            <button
              className={`tab-button ${activeTab === 'year-review' ? 'active' : ''}`}
              onClick={() => setActiveTab('year-review')}
            >
              Year in Review
            </button>
            <button
              className={`tab-button ${activeTab === 'topics' ? 'active' : ''}`}
              onClick={() => setActiveTab('topics')}
            >
              Topics
            </button>
            <button
              className={`tab-button ${activeTab === 'papers' ? 'active' : ''}`}
              onClick={() => setActiveTab('papers')}
            >
              Papers
            </button>
          </div>
        </div>
      </header>

      <main className="main-content">
        {activeTab === 'year-review' && <YearInReview />}
        {activeTab === 'topics' && <TopicsList />}
        {activeTab === 'papers' && <PapersList />}
      </main>
    </div>
  )
}

export default App
