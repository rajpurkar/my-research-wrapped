import { useState, useEffect, useRef } from 'react'

interface Author {
  full_name: string
  normalized_name: string
}

interface Paper {
  title: string
  summary: string
  weight: number
  role: string
  file_path: string
  authors: Author[]
}

interface Topic {
  name: string
  synthesis: string
  papers: Paper[]
}

interface NarrativeData {
  author: string
  introduction: string
  topics: Topic[]
}

function App() {
  const [narrative, setNarrative] = useState<NarrativeData | null>(null)
  const topicsRef = useRef<HTMLDivElement>(null)
  const [selectedPaper, setSelectedPaper] = useState<{topicIndex: number, paperIndex: number} | null>(null)

  useEffect(() => {
    // Load narrative data
    const loadNarrative = async () => {
      try {
        const narrativeData = await import('./outputs/narrative.json')
        setNarrative(narrativeData.default)
      } catch (error) {
        console.error('Error loading narrative data:', error)
      }
    }

    loadNarrative()
  }, [])

  const scrollToTopic = (index: number) => {
    const element = document.getElementById(`topic-${index}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // Generate more stars
  const generateStars = () => {
    const stars = [];
    for (let i = 0; i < 12; i++) {
      const delay = Math.random() * 4;
      const top = Math.random() * 100;
      const left = Math.random() * 100;
      stars.push(
        <div
          key={i}
          className="star"
          style={{
            top: `${top}%`,
            left: `${left}%`,
            animationDelay: `${delay}s`
          }}
        />
      );
    }
    return stars;
  };

  // Generate sparkles
  const generateSparkles = () => {
    const sparkles = [];
    for (let i = 0; i < 20; i++) {
      const delay = Math.random() * 4;
      const top = Math.random() * 100;
      const left = Math.random() * 100;
      const tx = (Math.random() - 0.5) * 100; // Random x translation
      const ty = (Math.random() - 0.5) * 100; // Random y translation
      sparkles.push(
        <div
          key={`sparkle-${i}`}
          className="sparkle"
          style={{
            top: `${top}%`,
            left: `${left}%`,
            animationDelay: `${delay}s`,
            '--tx': `${tx}px`,
            '--ty': `${ty}px`
          } as React.CSSProperties}
        />
      );
    }
    return sparkles;
  };

  if (!narrative) {
    return (
      <div className="loading-screen">
        <div className="loading-content">
          <div className="loading-spinner"></div>
          <p>Analyzing research impact...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>ResearchYearWrapped 🎁</h1>
          <div className="header-meta">
            <span className="year-tag">2024</span>
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* Hero Section */}
        <section className="hero-section">
          <div className="stars-container">
            {generateStars()}
            {generateSparkles()}
          </div>
          <div className="hero-content">
            <h1>My Research Wrapped 2024</h1>
            <h2 className="author-name">by {narrative.author}</h2>
            
            {/* Topics Grid */}
            <div className="topics-highlight">
              {narrative.topics.map((topic, index) => (
                <button
                  key={index}
                  className="topic-highlight-item"
                  onClick={() => scrollToTopic(index)}
                >
                  <span className="topic-number">0{index + 1}</span>
                  <span className="topic-name">{topic.name}</span>
                </button>
              ))}
            </div>

            <div className="narrative-text">
              {narrative.introduction.split('\n').map((paragraph, index) => (
                paragraph.trim() && (
                  <p key={`intro-${index}`} className="narrative-paragraph">
                    {paragraph}
                  </p>
                )
              ))}
            </div>
          </div>
        </section>

        {/* Research Areas Grid */}
        <section className="research-areas">
          <div className="section-header">
            <h2>Research Areas</h2>
            <p className="section-subtitle">Key themes and contributions from the year</p>
          </div>
          
          <div className="topics-grid" ref={topicsRef}>
            {narrative.topics.map((topic, index) => (
              <div id={`topic-${index}`} key={index} className="topic-group">
                <div className={`topic-card topic-variant-${index % 4}`}>
                  <div className="topic-content">
                    <div className="topic-header">
                      <span className="topic-number">0{index + 1}</span>
                      <h3>{topic.name}</h3>
                    </div>
                    <p className="topic-synthesis">{topic.synthesis}</p>
                  </div>
                </div>

                <div className="papers-grid">
                  {topic.papers.map((paper, pIndex) => (
                    <div
                      key={`${index}-paper-${pIndex}`}
                      className={`paper-item ${selectedPaper?.topicIndex === index && selectedPaper?.paperIndex === pIndex ? 'active' : ''}`}
                      onClick={() => setSelectedPaper({ topicIndex: index, paperIndex: pIndex })}
                    >
                      <div className="paper-item-content">
                        <h4 className="paper-title">{paper.title}</h4>
                        <p className="paper-summary">{paper.summary}</p>
                        <div className="paper-authors">
                          {paper.authors.map((author, aIndex) => (
                            <span key={aIndex} className="author-tag">{author.full_name}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
