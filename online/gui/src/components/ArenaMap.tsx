import { useEffect, useRef } from 'react'
import type { Snapshot } from '../types'

type Props = {
  snapshot: Snapshot
}

function drawRobot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  yaw: number,
  color: string,
  label: string,
) {
  const radius = 10
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.fill()
  ctx.strokeStyle = '#08111f'
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(x, y)
  ctx.lineTo(x + Math.cos(yaw) * 18, y + Math.sin(yaw) * 18)
  ctx.stroke()
  ctx.fillStyle = '#f8fafc'
  ctx.font = '12px sans-serif'
  ctx.fillText(label, x + 14, y - 12)
}

export default function ArenaMap({ snapshot }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const width = canvas.width
    const height = canvas.height
    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = '#07111f'
    ctx.fillRect(0, 0, width, height)

    const corners = snapshot.calibration.arena_corners_world
    const xs = corners.map((corner) => corner[0])
    const ys = corners.map((corner) => corner[1])
    const minX = Math.min(...xs, -1.12)
    const maxX = Math.max(...xs, 1.12)
    const minY = Math.min(...ys, -0.51)
    const maxY = Math.max(...ys, 0.51)
    const padding = 30
    const scaleX = (width - padding * 2) / (maxX - minX || 1)
    const scaleY = (height - padding * 2) / (maxY - minY || 1)
    const scale = Math.min(scaleX, scaleY)
    const toCanvas = (x: number, y: number): [number, number] => [
      padding + (x - minX) * scale,
      padding + (y - minY) * scale,
    ]

    const [x0, y0] = toCanvas(minX, minY)
    const [x1, y1] = toCanvas(maxX, maxY)
    ctx.strokeStyle = snapshot.world.ready ? '#35d399' : '#f97316'
    ctx.lineWidth = 2
    ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0))

    snapshot.world.obstacles.forEach((obstacle) => {
      const [x, y] = toCanvas(obstacle.pose.x, obstacle.pose.y)
      const size = obstacle.size_m * scale
      ctx.save()
      ctx.translate(x, y)
      ctx.rotate(-obstacle.pose.yaw)
      ctx.fillStyle = obstacle.visible ? 'rgba(56, 189, 248, 0.85)' : 'rgba(56, 189, 248, 0.35)'
      ctx.fillRect(-size / 2, -size / 2, size, size)
      ctx.restore()
      ctx.fillStyle = '#dbeafe'
      ctx.font = '11px sans-serif'
      ctx.fillText(`obs ${obstacle.tag_id}`, x + 8, y + 4)
    })

    const chaser = snapshot.world.chaser?.filtered_pose
    if (chaser) {
      const [x, y] = toCanvas(chaser.x, chaser.y)
      drawRobot(ctx, x, y, chaser.yaw, '#fb7185', 'chaser')
    }

    const evader = snapshot.world.evader?.filtered_pose
    if (evader) {
      const [x, y] = toCanvas(evader.x, evader.y)
      drawRobot(ctx, x, y, evader.yaw, '#fbbf24', 'evader')
    }
  }, [snapshot])

  return <canvas ref={ref} width={520} height={320} className="arena-canvas" />
}
