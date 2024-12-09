import { useState } from 'react'

interface Author {
  full_name: string
  normalized_name: string
}

interface Paper {
  title: string
  summary: string
  weight: number
  authors: Author[]
}

interface Topic {
  synthesis: string
  paper_summaries: Paper[]
}

const topicModules = import.meta.glob('../outputs/topics/*.json', { eager: true })

export function TopicsList() {
  const [topics] = useState<Record<string, Topic>>(() => {
    const topicsData: Record<string, Topic> = {}
    
    Object.entries(topicModules).forEach(([path, module]) => {
      const filename = path.split('/').pop()?.replace('.json', '') || ''
      const topicName = filename.replace(/_/g, ' ')
      topicsData[topicName] = (module as any).default
    })

    return topicsData
  })

  if (Object.keys(topics).length === 0) {
    return <div className="error-message">No topics found</div>
  }

  return (
    <div className="topics-container">
      <h2>Your Research Areas</h2>
      <p className="topics-intro">My Research Wrapped has identified these key research themes in your work:</p>
      
      <div className="topics-list">
        {Object.entries(topics).map(([name, topic]) => (
          <div key={name} className="topic-card">
            <h2 className="topic-title">{name}</h2>
            
            <div className="topic-synthesis">
              <h3>Overview</h3>
              {topic.synthesis.split('\n\n').map((paragraph, idx) => (
                <p key={idx}>{paragraph}</p>
              ))}
            </div>

            <div className="papers-section">
              <h3>Key Papers</h3>
              <div className="papers-grid">
                {topic.paper_summaries.map((paper, idx) => (
                  <div key={idx} className="paper-card">
                    <h4>{paper.title}</h4>
                    <p className="paper-summary">{paper.summary}</p>
                    <div className="paper-authors">
                      <h5>Authors:</h5>
                      <p>{paper.authors.map(author => author.full_name).join(', ')}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
} 