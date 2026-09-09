interface Props {
  name: string;
  color: string;
  emoji: string;
  side: 'left' | 'right';
  isSpeaking: boolean;
  isActive: boolean;
}

/**
 * Animated SVG debater.
 * - Whole character gently bobs (always-on idle).
 * - Mouth (<rect>) scales vertically on a loop while `isSpeaking`.
 * - When `isActive` but not speaking, the character is highlighted with a glow ring.
 */
export default function Character({ name, color, emoji, side, isSpeaking, isActive }: Props) {
  const flip = side === 'right' ? 'scale-x-[-1]' : '';

  return (
    <div className="flex flex-col items-center gap-3">
      <div
        className={`relative animate-bob ${flip}`}
        style={{ filter: isActive ? `drop-shadow(0 0 24px ${color}88)` : 'none' }}
      >
        <svg width="180" height="220" viewBox="0 0 180 220" xmlns="http://www.w3.org/2000/svg">
          {/* Body */}
          <ellipse cx="90" cy="180" rx="60" ry="30" fill={color} opacity="0.85" />
          <rect x="45" y="110" width="90" height="70" rx="18" fill={color} opacity="0.95" />
          {/* Head */}
          <circle cx="90" cy="75" r="50" fill={color} />
          <circle cx="90" cy="75" r="50" fill="white" opacity="0.12" />
          {/* Eyes */}
          <circle cx="72" cy="68" r="6" fill="#0f172a" />
          <circle cx="108" cy="68" r="6" fill="#0f172a" />
          <circle cx="74" cy="66" r="2" fill="white" />
          <circle cx="110" cy="66" r="2" fill="white" />
          {/* Cheeks */}
          <circle cx="62" cy="88" r="5" fill="#f472b6" opacity="0.55" />
          <circle cx="118" cy="88" r="5" fill="#f472b6" opacity="0.55" />
          {/* Mouth — animates via mouth-talk keyframe while speaking */}
          <rect
            x="78"
            y="92"
            width="24"
            height="10"
            rx="5"
            fill="#0f172a"
            className={`mouth-path ${isSpeaking ? 'animate-mouth-talk' : ''}`}
            style={{ transform: isSpeaking ? undefined : 'scaleY(0.4)' }}
          />
        </svg>
      </div>
      <div
        className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold transition-all ${
          isActive ? 'scale-110' : 'opacity-70'
        }`}
        style={{ backgroundColor: `${color}33`, color }}
      >
        <span>{emoji}</span>
        <span>{name}</span>
      </div>
    </div>
  );
}
