import { useState } from 'react'
import { FiLink, FiRefreshCw, FiTrendingUp, FiAlertCircle, FiHelpCircle } from 'react-icons/fi'
import { BarChart, Bar, ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import './App.css'

type AppState = 'entry' | 'processing' | 'results'

interface ProcessingStep {
  id: string
  label: string
  description: string
  status: 'pending' | 'active' | 'completed'
}

interface Cluster {
  title: string
  summary: string
  commentCount: number
}

interface Insight {
  text: string
}

interface Opportunity {
  type: 'feature' | 'complaint' | 'question'
  text: string
}

function App() {
  const [state, setState] = useState<AppState>('entry')
  const [url, setUrl] = useState('')
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([
    { id: '1', label: 'Fetching comments', description: 'This may take 2–5 minutes.', status: 'pending' },
    { id: '2', label: 'Creating embeddings', description: 'Organizing conversations semantically.', status: 'pending' },
    { id: '3', label: 'Clustering discussions', description: 'Grouping similar ideas together.', status: 'pending' },
    { id: '4', label: 'Filtering spam & noise', description: 'Removing low-quality and repetitive content.', status: 'pending' },
    { id: '5', label: 'Generating insights', description: 'Summarizing what shows up the most.', status: 'pending' },
  ])

  // State for dashboard data
  const [dashboardData, setDashboardData] = useState<any>(null)

  // Update the data based on dashboardData when available
  // Process actual data from dashboard
  const clusters: Cluster[] = dashboardData && dashboardData.dashboard_data?.focus_topics ?
    (dashboardData.dashboard_data?.focus_topics || []).map((topic: any) => ({
      title: topic.label || topic.topic_label || `Topic ${topic.cluster_id}`,
      summary: topic.summary || 'No summary available',
      commentCount: topic.comment_count || topic.count || 0
    })) : [
      {
        title: 'Video quality and production',
        summary: 'Viewers appreciate the clear visuals and professional editing, with many noting the improved camera work compared to earlier content.',
        commentCount: 342
      },
      {
        title: 'Topic depth and explanation',
        summary: 'Comments highlight that the explanation was thorough but some wanted more examples or follow-up details on specific points.',
        commentCount: 298
      },
      {
        title: 'Pacing and duration',
        summary: 'Mixed feedback on video length, with some viewers wanting longer, more detailed coverage while others preferred the concise format.',
        commentCount: 187
      },
      {
        title: 'Technical accuracy',
        summary: 'High praise for the technical depth and accuracy, with requests for more advanced tutorials on related topics.',
        commentCount: 156
      }
    ]

  const insights: Insight[] = dashboardData && dashboardData.dashboard_data?.emotions ?
    (dashboardData.dashboard_data?.emotions || [])
      .filter((emotion: any) => emotion.summary && emotion.summary.trim() !== '') // Only include emotions with actual summaries
      .map((emotion: any) => ({
        text: `${emotion.emotion} (${emotion.percentage}%): ${emotion.summary.substring(0, 200)}${emotion.summary.length > 200 ? '...' : ''}`
      })) : [
      { text: 'High engagement around practical examples and real-world applications' },
      { text: 'Strong positive sentiment toward educational value and clarity' },
      { text: 'Requests for follow-up content on advanced topics' },
      { text: 'Appreciation for timestamps and chapter markers' }
    ]

  const opportunities: Opportunity[] = dashboardData && dashboardData.dashboard_data?.focus_topics ?
    (dashboardData.dashboard_data?.focus_topics || []).map((topic: any) => ({
      type: 'feature', // Default type, could be determined from topic data
      text: `${topic.label || `Topic ${topic.cluster_id}`}: ${topic.summary.substring(0, 150)}${topic.summary.length > 150 ? '...' : ''}`
    })) : [
      { type: 'feature', text: 'Create a series covering advanced techniques' },
      { type: 'question', text: 'How to handle edge cases in this scenario?' },
      { type: 'complaint', text: 'Audio quality could be improved in outdoor segments' },
      { type: 'feature', text: 'Add downloadable resources or templates' }
    ]

  // Use actual sentiment distribution data from the dashboard if available
  const clusterChartData = dashboardData && dashboardData.dashboard_data?.summary_stats?.theme_sentiment_stats ?
    (dashboardData.dashboard_data?.summary_stats.theme_sentiment_stats || []).map((theme: any) => ({
      name: theme.label || `Theme ${theme.cluster_id}`,
      value: theme.comment_count,
    })) : clusters.map(cluster => ({
      name: cluster.title.split(' ').slice(0, 2).join(' '),
      value: cluster.commentCount,
    }))

  const handleAnalyze = async () => {
    if (!url.trim()) return

    // Extract video ID from URL
    const videoId = extractVideoId(url)
    if (!videoId) {
      alert('Invalid YouTube URL')
      return
    }

    setState('processing')
    await triggerAnalysis(videoId)
  }

  const extractVideoId = (url: string): string | null => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/
    const match = url.match(regExp)
    return (match && match[2].length === 11) ? match[2] : null
  }

  const triggerAnalysis = async (videoId: string) => {
    try {
      // For demo purposes, skip the actual analysis and go directly to results
      // using the pre-existing analysis data
      setProcessingSteps(prev => prev.map((step, i) =>
        i === 0 ? { ...step, status: 'active' } : step
      ))

      // Simulate the processing steps quickly for demo
      for (let i = 0; i < processingSteps.length; i++) {
        setProcessingSteps(prev => prev.map((step, idx) =>
          idx === i ? { ...step, status: 'active' } : step
        ))
        await new Promise(resolve => setTimeout(resolve, 500)) // Fast demo
        setProcessingSteps(prev => prev.map((step, idx) =>
          idx === i ? { ...step, status: 'completed' } : step
        ))
      }

      // Load the demo data directly
      await fetchDashboardData(videoId)
    } catch (error) {
      console.error('Analysis failed:', error)
      alert('Analysis failed: ' + (error as Error).message)
      setState('entry')
    }
  }

  const fetchDashboardData = async (videoId: string) => {
    try {
      // For demo purposes, load the actual JSON file from the analysis results
      // In a real application, this would fetch from the API as before
      const response = await fetch('http://localhost:8000/api/dashboard/8bMh8azh3CY/')
      if (!response.ok) {
        console.warn('API fetch failed, falling back to demo data:', response.statusText)

        // Load demo data from the actual analysis results file
        const demoResponse = await fetch('/demo-analysis.json')
        if (demoResponse.ok) {
          const demoData = await demoResponse.json()
          setDashboardData({
            video_id: '8bMh8azh3CY',
            dashboard_data: demoData,
            status: 'success'
          })
          setState('results')
          return
        } else {
          throw new Error('Demo data also failed to load')
        }
      }

      const data = await response.json()
      // Set the dashboard data to update the UI
      setDashboardData(data)
      setState('results')
    } catch (error) {
      console.error('Failed to fetch dashboard:', error)
      // Even if API fails, try to load demo data
      try {
        const demoResponse = await fetch('/demo-analysis.json')
        if (demoResponse.ok) {
          const demoData = await demoResponse.json()
          setDashboardData({
            video_id: '8bMh8azh3CY',
            dashboard_data: demoData,
            status: 'success'
          })
          setState('results')
        } else {
          throw new Error('Demo data also failed to load')
        }
      } catch (demoError) {
        console.error('Failed to load demo data:', demoError)
        alert('Failed to fetch dashboard: ' + (error as Error).message)
      }
    }
  }

  const handleNewAnalysis = () => {
    setState('entry')
    setUrl('')
    setProcessingSteps(prev => prev.map(step => ({ ...step, status: 'pending' })))
  }

  return (
    <div className="min-h-screen bg-gradient-dark">
      <header className="sticky top-0 z-50 glass border-b border-white/5">
        <div className="max-w-7xl mx-auto w-full px-6 py-4 flex items-center justify-between">
          <h1 className="text-lg font-medium text-text-primary">AI Comments Analyzer</h1>
          {state === 'results' && (
            <button
              onClick={handleNewAnalysis}
              className="px-4 py-2 text-text-secondary hover:text-text-primary transition-colors flex items-center gap-2 text-sm font-medium rounded-lg hover:bg-white/5"
            >
              <FiRefreshCw size={16} />
              New analysis
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 px-6 py-8">
        <div className="max-w-7xl mx-auto">
          {state === 'entry' && (
            <div className="flex items-center justify-center min-h-[calc(100vh-120px)]">
              <div className="w-full max-w-2xl animate-in fade-in duration-300">
                <div className="glass rounded-3xl p-12 shadow-glass">
                  <div className="space-y-8">
                    <div className="space-y-4">
                      <label className="block">
                        <div className="relative">
                          <FiLink className="absolute left-5 top-1/2 -translate-y-1/2 text-text-tertiary pointer-events-none" size={20} />
                          <input
                            type="text"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                            placeholder="Paste a YouTube video or channel link"
                            className="w-full pl-14 pr-5 py-5 glass-light border border-white/10 rounded-2xl bg-white/5 text-text-primary placeholder-text-tertiary focus:border-dark-accent focus:ring-2 focus:ring-dark-accent/30 transition-all outline-none text-base focus:bg-white/10"
                          />
                        </div>
                      </label>
      </div>

                    <button
                      onClick={handleAnalyze}
                      disabled={!url.trim()}
                      className="w-full bg-gradient-accent hover:opacity-90 text-white py-5 px-6 rounded-2xl font-medium text-base transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-glow-purple disabled:shadow-none active:scale-[0.98]"
                    >
                      Analyze comments
        </button>

                    <p className="text-center text-text-secondary text-sm">
                      We'll analyze public comments to surface trends and ideas.
        </p>
      </div>
                </div>
              </div>
            </div>
          )}

          {state === 'processing' && (
            <div className="flex items-center justify-center min-h-[calc(100vh-120px)]">
              <div className="w-full max-w-2xl animate-in fade-in duration-300">
                <div className="glass rounded-3xl p-12 shadow-glass">
                  <div className="space-y-8">
                    {processingSteps.map((step) => (
                      <div
                        key={step.id}
                        className={`flex items-start gap-4 transition-all duration-500 ease-out ${
                          step.status === 'pending' ? 'opacity-30' : 'opacity-100'
                        }`}
                      >
                        <div className="flex-shrink-0 mt-1">
                          {step.status === 'completed' && (
                            <div className="w-6 h-6 rounded-full bg-gradient-accent flex items-center justify-center shadow-glow-purple">
                              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                              </svg>
                            </div>
                          )}
                          {step.status === 'active' && (
                            <div className="w-6 h-6 flex items-center justify-center">
                              <div className="flex gap-1">
                                <div className="w-1.5 h-1.5 rounded-full bg-dark-accent-light opacity-60 animate-bounce" style={{ animationDelay: '0ms', animationDuration: '1.4s' }}></div>
                                <div className="w-1.5 h-1.5 rounded-full bg-dark-accent-light opacity-60 animate-bounce" style={{ animationDelay: '200ms', animationDuration: '1.4s' }}></div>
                                <div className="w-1.5 h-1.5 rounded-full bg-dark-accent-light opacity-60 animate-bounce" style={{ animationDelay: '400ms', animationDuration: '1.4s' }}></div>
                              </div>
                            </div>
                          )}
                          {step.status === 'pending' && (
                            <div className="w-6 h-6 rounded-full border-2 border-text-tertiary/30"></div>
                          )}
                        </div>

                        <div className="flex-1">
                          <h3 className={`text-base font-medium mb-1 ${
                            step.status === 'active' ? 'text-dark-accent-light' : 'text-text-primary'
                          }`}>
                            {step.label}
                          </h3>
                          <p className="text-sm text-text-secondary">
                            {step.description}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {state === 'results' && (
            <div className="space-y-8 animate-in fade-in duration-500">
              <section className="relative glass rounded-3xl p-8 shadow-glass overflow-hidden">
                <div className="absolute inset-0 bg-gradient-hero opacity-20"></div>
                <div className="relative z-10">
                  <div className="mb-6">
                    <h2 className="text-2xl font-semibold text-text-primary mb-2">Audience signals at a glance</h2>
                    <p className="text-text-secondary text-sm">Cluster distribution across comment themes</p>
                  </div>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={clusterChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
                        <XAxis 
                          dataKey="name" 
                          stroke="#71717a"
                          tick={{ fill: '#a1a1aa', fontSize: 12 }}
                          axisLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
                        />
                        <YAxis 
                          stroke="#71717a"
                          tick={{ fill: '#a1a1aa', fontSize: 12 }}
                          axisLine={{ stroke: 'rgba(255, 255, 255, 0.1)' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(26, 26, 36, 0.95)', 
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            borderRadius: '12px',
                            color: '#f5f5f7'
                          }}
                          cursor={{ fill: 'rgba(99, 102, 241, 0.1)' }}
                        />
                        <Bar 
                          dataKey="value" 
                          fill="url(#colorGradient)"
                          radius={[8, 8, 0, 0]}
                        />
                        <defs>
                          <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#818cf8" stopOpacity={0.8}/>
                            <stop offset="100%" stopColor="#6366f1" stopOpacity={0.9}/>
                          </linearGradient>
                        </defs>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </section>

              <section>
                <h2 className="text-2xl font-semibold text-text-primary mb-6">Discussion clusters</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {clusters.map((cluster, index) => (
                    <div
                      key={index}
                      className="glass rounded-2xl p-6 shadow-glass hover:shadow-glow-blue/20 transition-all duration-300 border border-white/5 hover:border-white/10 group"
                    >
                      <h3 className="text-lg font-semibold text-text-primary mb-2 group-hover:text-gradient transition-colors">
                        {cluster.title}
                      </h3>
                      <p className="text-text-secondary mb-4 leading-relaxed text-sm">{cluster.summary}</p>
                      <p className="text-xs text-text-tertiary">~{cluster.commentCount} comments</p>
                    </div>
                  ))}
                </div>
              </section>

              <section>
                <h2 className="text-2xl font-semibold text-text-primary mb-6">Common ideas & themes</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {insights.map((insight, index) => (
                    <div
                      key={index}
                      className="glass-light rounded-xl p-4 shadow-glass-sm hover:shadow-glass transition-all duration-200 border border-white/5 hover:border-white/10 group flex items-start gap-3"
                    >
                      <div className="w-1.5 h-1.5 rounded-full bg-dark-accent mt-2 flex-shrink-0"></div>
                      <p className="text-text-primary leading-relaxed text-sm group-hover:text-text-primary">{insight.text}</p>
                    </div>
                  ))}
                </div>
              </section>

              <section>
                <h2 className="text-2xl font-semibold text-text-primary mb-6">Opportunities & signals</h2>
                <div className="space-y-3">
                  {opportunities.map((opp, index) => (
                    <div
                      key={index}
                      className={`glass rounded-xl p-4 shadow-glass-sm hover:shadow-glass transition-all duration-200 border-l-4 ${
                        opp.type === 'feature' 
                          ? 'border-glow-blue hover:shadow-glow-blue/20' 
                          : opp.type === 'complaint' 
                          ? 'border-orange-400/60' 
                          : 'border-blue-400/60'
                      } border-t border-r border-b border-white/5 hover:border-white/10 flex items-start gap-3`}
                    >
                      <div className="mt-0.5">
                        {opp.type === 'feature' && <FiTrendingUp className="text-glow-blue" size={18} />}
                        {opp.type === 'complaint' && <FiAlertCircle className="text-orange-400" size={18} />}
                        {opp.type === 'question' && <FiHelpCircle className="text-blue-400" size={18} />}
                      </div>
                      <p className="text-text-primary leading-relaxed text-sm flex-1">{opp.text}</p>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
