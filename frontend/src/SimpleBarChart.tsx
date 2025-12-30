interface ChartData {
  name: string
  value: number
}

interface SimpleBarChartProps {
  data: ChartData[]
}

export function SimpleBarChart({ data }: SimpleBarChartProps) {
  const maxValue = Math.max(...data.map(d => d.value))
  const chartHeight = 180
  const chartWidth = 800
  const paddingLeft = 60
  const paddingRight = 40
  const paddingTop = 20
  const paddingBottom = 50
  const availableWidth = chartWidth - paddingLeft - paddingRight
  const barWidth = (availableWidth / data.length) * 0.7
  const spacing = (availableWidth / data.length) * 0.3

  return (
    <div className="h-64 w-full overflow-x-auto">
      <svg 
        viewBox={`0 0 ${chartWidth} ${chartHeight}`} 
        className="w-full h-full" 
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#818cf8" stopOpacity="0.8"/>
            <stop offset="100%" stopColor="#6366f1" stopOpacity="0.9"/>
          </linearGradient>
        </defs>
        
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = paddingTop + ratio * (chartHeight - paddingTop - paddingBottom)
          return (
            <line
              key={ratio}
              x1={paddingLeft}
              y1={y}
              x2={chartWidth - paddingRight}
              y2={y}
              stroke="rgba(255, 255, 255, 0.05)"
              strokeWidth="1"
            />
          )
        })}
        
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const value = Math.round(maxValue * (1 - ratio))
          const y = paddingTop + ratio * (chartHeight - paddingTop - paddingBottom)
          return (
            <text
              key={ratio}
              x={paddingLeft - 10}
              y={y + 4}
              textAnchor="end"
              fill="#71717a"
              fontSize="12"
            >
              {value}
            </text>
          )
        })}
        
        {data.map((item, index) => {
          const barHeight = (item.value / maxValue) * (chartHeight - paddingTop - paddingBottom)
          const x = paddingLeft + index * (barWidth + spacing) + spacing / 2
          const y = chartHeight - paddingBottom - barHeight
          
          return (
            <g key={index} className="group">
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                fill="url(#barGradient)"
                rx="4"
                className="transition-all duration-500 hover:opacity-80 cursor-pointer"
              />
              <text
                x={x + barWidth / 2}
                y={y - 8}
                textAnchor="middle"
                fill="#f5f5f7"
                fontSize="12"
                className="opacity-0 group-hover:opacity-100 transition-opacity font-medium pointer-events-none"
              >
                {item.value}
              </text>
              <text
                x={x + barWidth / 2}
                y={chartHeight - paddingBottom + 20}
                textAnchor="middle"
                fill="#a1a1aa"
                fontSize="11"
                className="font-medium"
              >
                {item.name}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

