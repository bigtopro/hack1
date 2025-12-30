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
  // Process actual data from dashboard - API returns data directly, not nested
  const clusters: Cluster[] = dashboardData && dashboardData.focus_topics ?
    (dashboardData.focus_topics || []).map((topic: any) => ({
      title: topic.label || `Topic ${topic.cluster_id}`,
      summary: topic.summary || 'No summary available',
      commentCount: topic.comment_count || 0
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

  const insights: Insight[] = dashboardData && dashboardData.emotions ?
    (dashboardData.emotions || [])
      .filter((emotion: any) => emotion.summary && emotion.summary.trim() !== '') // Only include emotions with actual summaries
      .slice(0, 8) // Show top 8 emotions
      .map((emotion: any) => ({
        text: `${emotion.emotion.charAt(0).toUpperCase() + emotion.emotion.slice(1)} (${emotion.percentage}%): ${emotion.summary.substring(0, 200)}${emotion.summary.length > 200 ? '...' : ''}`
      })) : [
      { text: 'High engagement around practical examples and real-world applications' },
      { text: 'Strong positive sentiment toward educational value and clarity' },
      { text: 'Requests for follow-up content on advanced topics' },
      { text: 'Appreciation for timestamps and chapter markers' }
    ]

  const opportunities: Opportunity[] = dashboardData && dashboardData.focus_topics ?
    (dashboardData.focus_topics || []).slice(0, 6).map((topic: any) => {
      // Determine type based on dominant sentiment
      const dominantSentiment = topic.sentiment_distribution ? 
        Object.entries(topic.sentiment_distribution).sort((a: any, b: any) => b[1] - a[1])[0]?.[0] : 'neutral'
      let type: 'feature' | 'complaint' | 'question' = 'feature'
      if (['anger', 'disapproval', 'annoyance', 'disappointment'].includes(dominantSentiment)) {
        type = 'complaint'
      } else if (['curiosity', 'confusion'].includes(dominantSentiment)) {
        type = 'question'
      }
      return {
        type,
        text: `${topic.label || `Topic ${topic.cluster_id}`}: ${topic.summary ? topic.summary.substring(0, 150) : 'No summary available'}${topic.summary && topic.summary.length > 150 ? '...' : ''}`
      }
    }) : [
      { type: 'feature', text: 'Create a series covering advanced techniques' },
      { type: 'question', text: 'How to handle edge cases in this scenario?' },
      { type: 'complaint', text: 'Audio quality could be improved in outdoor segments' },
      { type: 'feature', text: 'Add downloadable resources or templates' }
    ]

  // Use actual theme data from focus_topics for better labels, or fallback to theme_sentiment_stats
  const clusterChartData = dashboardData && dashboardData.focus_topics ?
    (dashboardData.focus_topics || [])
      .sort((a: any, b: any) => (b.comment_count || 0) - (a.comment_count || 0)) // Sort by comment count
      .slice(0, 10) // Show top 10 themes
      .map((topic: any) => {
        // Use the label, but clean it up if it's too long or generic
        let label = topic.label || `Topic ${topic.cluster_id}`;
        
        // Clean up labels that start with "The topic is mainly about"
        if (label.toLowerCase().startsWith("the topic is mainly about")) {
          // Extract the meaningful part after "about"
          const aboutIndex = label.toLowerCase().indexOf("about");
          if (aboutIndex !== -1) {
            const meaningfulPart = label.substring(aboutIndex + 6).trim();
            // Capitalize first letter
            label = meaningfulPart.charAt(0).toUpperCase() + meaningfulPart.slice(1);
          }
        }
        
        // If label is still generic like "Topic", try to get from summary
        if (label === "Topic" || label.length < 3) {
          if (topic.summary && topic.summary.length > 0) {
            // Extract first meaningful sentence or phrase
            const firstSentence = topic.summary.split('.')[0] || topic.summary.split('\n')[0];
            const words = firstSentence.split(' ').slice(0, 5).join(' ');
            label = words.length > 25 ? words.substring(0, 22) + '...' : words;
          } else {
            label = `Topic ${topic.cluster_id}`;
          }
        }
        
        // Don't truncate - let the chart handle it with rotation
        // Keep labels descriptive but reasonable length
        if (label.length > 30) {
          label = label.substring(0, 27) + '...';
        }
        
        return {
          name: label,
          value: topic.comment_count || 0,
          fullLabel: topic.label || `Topic ${topic.cluster_id}`, // Keep full label for tooltip
        };
      }) : dashboardData && dashboardData.summary_stats?.theme_sentiment_stats ?
    (dashboardData.summary_stats.theme_sentiment_stats || [])
      .slice(0, 10) // Show top 10 themes
      .map((theme: any) => ({
        name: theme.label || `Theme ${theme.cluster_id}`,
        value: theme.comment_count,
        fullLabel: theme.label || `Theme ${theme.cluster_id}`,
      })) : clusters.map(cluster => ({
      name: cluster.title.split(' ').slice(0, 2).join(' '),
      value: cluster.commentCount,
      fullLabel: cluster.title,
    }))

  const handleAnalyze = async () => {
    if (!url.trim()) return

    // Extract video ID from URL (for validation, but using hardcoded data)
    const videoId = extractVideoId(url)
    if (!videoId) {
      alert('Invalid YouTube URL')
      return
    }

    setState('processing')
    await triggerAnalysis()
  }

  const extractVideoId = (url: string): string | null => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/
    const match = url.match(regExp)
    return (match && match[2].length === 11) ? match[2] : null
  }

  const triggerAnalysis = async () => {
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

      // Load the hardcoded analysis data directly
      await fetchDashboardData()
    } catch (error) {
      console.error('Analysis failed:', error)
      alert('Analysis failed: ' + (error as Error).message)
      setState('entry')
    }
  }

  const fetchDashboardData = async () => {
    try {
      // Load hardcoded analysis data from the JSON file
      const response = await fetch('/demo-analysis.json')
      if (!response.ok) {
        throw new Error('Failed to load analysis data')
      }

      const data = await response.json()
      // Set the dashboard data directly - hardcoded from analysis_dashboard_20251231_002010.json
      setDashboardData(data)
      setState('results')
    } catch (error) {
      console.error('Failed to load dashboard:', error)
      alert('Failed to load dashboard: ' + (error as Error).message)
      setState('entry')
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
              {dashboardData && dashboardData.meta && (
                <div className="glass rounded-2xl p-6 shadow-lg border-2 border-purple-500/30 bg-gradient-to-r from-purple-500/10 to-pink-500/10">
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-purple-300 text-sm font-medium">Video ID:</span>
                        <span className="text-white font-bold text-base">{dashboardData.meta.video_id}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-cyan-300 text-sm font-medium">Total Comments:</span>
                        <span className="text-white font-bold text-lg">{dashboardData.meta.total_comments?.toLocaleString() || 'N/A'}</span>
                      </div>
                    </div>
                    {dashboardData.meta.analysis_timestamp && (
                      <div className="text-right">
                        <p className="text-pink-300 text-xs font-medium mb-1">Analyzed</p>
                        <p className="text-white text-sm font-semibold">
                          {new Date(dashboardData.meta.analysis_timestamp).toLocaleString()}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}
              <section className="relative glass rounded-3xl p-8 shadow-glass overflow-hidden border-2 border-blue-500/20">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10"></div>
                <div className="relative z-10">
                  <div className="mb-6">
                    <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 mb-2">Audience Signals at a Glance</h2>
                    <p className="text-text-secondary text-base">Cluster distribution across comment themes</p>
                  </div>
                  <div className="h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={clusterChartData} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                        <XAxis 
                          dataKey="name" 
                          stroke="#a78bfa"
                          tick={{ fill: '#e9d5ff', fontSize: 10, fontWeight: 500 }}
                          axisLine={{ stroke: '#a78bfa', strokeWidth: 2 }}
                          angle={-45}
                          textAnchor="end"
                          height={70}
                          interval={0}
                        />
                        <YAxis 
                          stroke="#a78bfa"
                          tick={{ fill: '#e9d5ff', fontSize: 11, fontWeight: 500 }}
                          axisLine={{ stroke: '#a78bfa', strokeWidth: 2 }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(30, 27, 75, 0.98)', 
                            border: '2px solid rgba(167, 139, 250, 0.5)',
                            borderRadius: '12px',
                            color: '#f5f5f7',
                            boxShadow: '0 10px 40px rgba(167, 139, 250, 0.3)',
                            padding: '12px'
                          }}
                          cursor={{ fill: 'rgba(167, 139, 250, 0.2)' }}
                          formatter={(value: any) => [`${value.toLocaleString()} comments`, '']}
                          labelFormatter={(label: string) => {
                            // Find the full label from the data
                            const dataPoint = clusterChartData.find((d: any) => d.name === label);
                            return dataPoint?.fullLabel || label;
                          }}
                        />
                        <Bar 
                          dataKey="value" 
                          fill="url(#colorGradient)"
                          radius={[12, 12, 0, 0]}
                          stroke="#a78bfa"
                          strokeWidth={1}
                        />
                        <defs>
                          <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#818cf8" stopOpacity={1}/>
                            <stop offset="50%" stopColor="#a78bfa" stopOpacity={1}/>
                            <stop offset="100%" stopColor="#ec4899" stopOpacity={1}/>
                          </linearGradient>
                        </defs>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </section>

              {dashboardData && dashboardData.summary_stats?.overall_sentiment?.raw && (
                <section className="glass rounded-3xl p-8 shadow-glass border-2 border-purple-500/20">
                  <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 mb-2">Overall Sentiment Distribution</h2>
                  <p className="text-text-secondary text-sm mb-6">Emotional breakdown of all comments</p>
                  
                  {/* Legend */}
                  <div className="mb-6 p-4 rounded-xl bg-gradient-to-r from-gray-800/50 to-gray-900/50 border border-white/10">
                    <p className="text-white font-semibold text-sm mb-3">Sentiment Categories:</p>
                    <div className="flex flex-wrap gap-4">
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded bg-gradient-to-br from-green-400 to-emerald-500 border-2 border-green-400/30"></div>
                        <span className="text-gray-200 text-xs font-medium">Positive Emotions</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded bg-gradient-to-br from-red-400 to-rose-500 border-2 border-red-400/30"></div>
                        <span className="text-gray-200 text-xs font-medium">Negative Emotions</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded bg-gradient-to-br from-blue-400 to-cyan-500 border-2 border-blue-400/30"></div>
                        <span className="text-gray-200 text-xs font-medium">Neutral Emotions</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                    {Object.entries(dashboardData.summary_stats.overall_sentiment.raw)
                      .filter(([_, value]: [string, any]) => value > 0.5) // Only show emotions > 0.5%
                      .sort(([_, a]: [string, any], [__, b]: [string, any]) => b - a)
                      .slice(0, 12)
                      .map(([emotion, percentage]: [string, any]) => {
                        // Color coding based on emotion type
                        const getEmotionColor = (emotion: string) => {
                          const positive = ['joy', 'love', 'admiration', 'gratitude', 'approval', 'optimism', 'excitement', 'amusement', 'caring'];
                          const negative = ['anger', 'sadness', 'disapproval', 'annoyance', 'disappointment', 'disgust', 'fear', 'remorse', 'embarrassment'];
                          
                          if (positive.includes(emotion)) {
                            return 'from-green-400 to-emerald-500 border-green-400/30';
                          } else if (negative.includes(emotion)) {
                            return 'from-red-400 to-rose-500 border-red-400/30';
                          } else {
                            // Neutral emotions: neutral, curiosity, surprise, confusion, realization, desire
                            return 'from-blue-400 to-cyan-500 border-blue-400/30';
                          }
                        };
                        
                        const colorClass = getEmotionColor(emotion);
                        
                        return (
                          <div 
                            key={emotion} 
                            className={`rounded-xl p-4 border-2 bg-gradient-to-br ${colorClass} shadow-lg hover:scale-105 transition-transform duration-200`}
                          >
                            <p className="text-white/90 text-xs font-medium mb-2 capitalize drop-shadow-sm">{emotion}</p>
                            <p className="text-white text-2xl font-bold drop-shadow-md">{percentage.toFixed(1)}%</p>
                          </div>
                        );
                      })}
                  </div>
                  <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-400/30">
                    <p className="text-text-secondary text-sm">
                      Dominant sentiment: <span className="text-white capitalize font-bold text-lg ml-2">
                        {dashboardData.summary_stats.overall_sentiment.dominant_raw}
                      </span>
                    </p>
                  </div>
                </section>
              )}

              <section>
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400">Discussion Clusters</h2>
                  <div className="hidden md:flex items-center gap-3 px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-400/30">
                    <p className="text-cyan-200 text-xs font-medium">Color Rotation:</p>
                    <div className="flex gap-1">
                      <div className="w-2 h-2 rounded-full bg-blue-400"></div>
                      <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                      <div className="w-2 h-2 rounded-full bg-indigo-400"></div>
                      <div className="w-2 h-2 rounded-full bg-pink-400"></div>
                      <div className="w-2 h-2 rounded-full bg-cyan-400"></div>
                      <div className="w-2 h-2 rounded-full bg-violet-400"></div>
                    </div>
                  </div>
                </div>
                
                {/* Cluster Legend */}
                <div className="mb-5 p-4 rounded-xl bg-gradient-to-r from-gray-800/50 to-gray-900/50 border border-white/10">
                  <p className="text-white font-semibold text-sm mb-3">Cluster Information:</p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></div>
                      <span className="text-gray-300">Each cluster represents a distinct discussion topic</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-blue-400"></div>
                      <span className="text-gray-300">Colors rotate to distinguish different themes</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-purple-400"></div>
                      <span className="text-gray-300">Comment count shows engagement level</span>
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {clusters.map((cluster, index) => {
                    const colors = [
                      'from-blue-500/20 to-cyan-500/20 border-blue-400/40',
                      'from-purple-500/20 to-pink-500/20 border-purple-400/40',
                      'from-indigo-500/20 to-blue-500/20 border-indigo-400/40',
                      'from-pink-500/20 to-rose-500/20 border-pink-400/40',
                      'from-cyan-500/20 to-teal-500/20 border-cyan-400/40',
                      'from-violet-500/20 to-purple-500/20 border-violet-400/40',
                    ];
                    const colorClass = colors[index % colors.length];
                    
                    return (
                      <div
                        key={index}
                        className={`glass rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border-2 bg-gradient-to-br ${colorClass} hover:scale-[1.02] group`}
                      >
                        <h3 className="text-xl font-bold text-white mb-3 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-cyan-300 group-hover:to-blue-300 transition-all">
                          {cluster.title}
                        </h3>
                        <p className="text-gray-200 mb-4 leading-relaxed text-sm">{cluster.summary}</p>
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></div>
                          <p className="text-cyan-300 text-sm font-semibold">~{cluster.commentCount.toLocaleString()} comments</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>

              <section>
                <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 via-orange-400 to-pink-400 mb-6">Common Ideas & Themes</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {insights.map((insight, index) => {
                    const accentColors = [
                      'bg-gradient-to-r from-yellow-400 to-orange-400',
                      'bg-gradient-to-r from-pink-400 to-rose-400',
                      'bg-gradient-to-r from-cyan-400 to-blue-400',
                      'bg-gradient-to-r from-purple-400 to-indigo-400',
                      'bg-gradient-to-r from-green-400 to-emerald-400',
                      'bg-gradient-to-r from-blue-400 to-cyan-400',
                      'bg-gradient-to-r from-orange-400 to-red-400',
                      'bg-gradient-to-r from-indigo-400 to-purple-400',
                    ];
                    const accentColor = accentColors[index % accentColors.length];
                    
                    return (
                      <div
                        key={index}
                        className="glass-light rounded-xl p-5 shadow-lg hover:shadow-xl transition-all duration-200 border-2 border-white/10 hover:border-white/20 bg-gradient-to-br from-white/5 to-white/10 hover:scale-[1.02] group flex items-start gap-4"
                      >
                        <div className={`w-3 h-3 rounded-full ${accentColor} mt-1.5 flex-shrink-0 shadow-lg animate-pulse`}></div>
                        <p className="text-gray-100 leading-relaxed text-sm font-medium group-hover:text-white transition-colors">{insight.text}</p>
                      </div>
                    );
                  })}
                </div>
              </section>

              <section>
                <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 mb-6">Opportunities & Signals</h2>
                
                {/* Opportunities Legend */}
                <div className="mb-5 p-4 rounded-xl bg-gradient-to-r from-gray-800/50 to-gray-900/50 border border-white/10">
                  <p className="text-white font-semibold text-sm mb-3">Signal Types:</p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded bg-gradient-to-br from-green-400 to-emerald-500 border-2 border-green-400/30 flex items-center justify-center">
                        <FiTrendingUp className="text-white" size={12} />
                      </div>
                      <span className="text-gray-300">Feature Opportunities</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded bg-gradient-to-br from-orange-400 to-red-500 border-2 border-orange-400/30 flex items-center justify-center">
                        <FiAlertCircle className="text-white" size={12} />
                      </div>
                      <span className="text-gray-300">Complaints & Concerns</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded bg-gradient-to-br from-blue-400 to-cyan-500 border-2 border-blue-400/30 flex items-center justify-center">
                        <FiHelpCircle className="text-white" size={12} />
                      </div>
                      <span className="text-gray-300">Questions & Curiosity</span>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  {opportunities.map((opp, index) => {
                    const typeStyles = {
                      feature: {
                        border: 'border-l-4 border-green-400',
                        bg: 'bg-gradient-to-r from-green-500/20 to-emerald-500/20',
                        icon: 'text-green-400',
                        iconBg: 'bg-green-400/20',
                      },
                      complaint: {
                        border: 'border-l-4 border-orange-400',
                        bg: 'bg-gradient-to-r from-orange-500/20 to-red-500/20',
                        icon: 'text-orange-400',
                        iconBg: 'bg-orange-400/20',
                      },
                      question: {
                        border: 'border-l-4 border-blue-400',
                        bg: 'bg-gradient-to-r from-blue-500/20 to-cyan-500/20',
                        icon: 'text-blue-400',
                        iconBg: 'bg-blue-400/20',
                      },
                    };
                    
                    const style = typeStyles[opp.type];
                    
                    return (
                      <div
                        key={index}
                        className={`glass rounded-xl p-5 shadow-lg hover:shadow-xl transition-all duration-200 ${style.border} ${style.bg} border-t border-r border-b border-white/10 hover:border-white/20 hover:scale-[1.01] flex items-start gap-4`}
                      >
                        <div className={`mt-0.5 p-2 rounded-lg ${style.iconBg}`}>
                          {opp.type === 'feature' && <FiTrendingUp className={style.icon} size={20} />}
                          {opp.type === 'complaint' && <FiAlertCircle className={style.icon} size={20} />}
                          {opp.type === 'question' && <FiHelpCircle className={style.icon} size={20} />}
                        </div>
                        <p className="text-gray-100 leading-relaxed text-sm font-medium flex-1">{opp.text}</p>
                      </div>
                    );
                  })}
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
