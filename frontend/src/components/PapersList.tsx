import { useState } from 'react'

interface Author {
  full_name: string
  normalized_name: string
}

interface Paper {
  title: string
  authors: Author[]
  summary: string
  weight: number
  role: string
  file_path: string
}

const paperModules = import.meta.glob('../outputs/papers/*.json', { eager: true })

export function PapersList() {
  const [papers] = useState<Record<string, Paper>>(() => {
    const papersData: Record<string, Paper> = {}
    
    Object.entries(paperModules).forEach(([path, module]) => {
      const filename = path.split('/').pop()?.replace('.json', '') || ''
      const paperName = filename.replace(/_/g, ' ')
      papersData[paperName] = (module as any).default
    })

    return papersData
  })

  const getGoogleScholarSearchUrl = (title: string) => {
    const encodedTitle = encodeURIComponent(title)
    return `https://scholar.google.com/scholar?q=${encodedTitle}`
  }

  if (Object.keys(papers).length === 0) {
    return <div className="error-message">No papers found</div>
  }

  return (
    <div className="content-container">
      <div className="papers-container">
        {Object.entries(papers).map(([id, paper]) => (
          <article key={id} className="paper-item">
            <header className="paper-header">
              <h2 className="paper-title">{paper.title}</h2>
            </header>

            <div className="paper-body">
              <div className="paper-summary">
                <h3>Summary</h3>
                <p>{paper.summary}</p>
              </div>

              <div className="paper-authors">
                <h3>Authors</h3>
                <div className="authors-list">
                  {paper.authors.map((author, idx) => (
                    <span key={idx} className="author-name">
                      {author.full_name}
                    </span>
                  ))}
                </div>
              </div>

              <div className="paper-link">
                <a 
                  href={getGoogleScholarSearchUrl(paper.title)}
                  target="_blank" 
                  rel="noopener noreferrer"
                >
                  Find Paper on Google Scholar
                </a>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  )
} 