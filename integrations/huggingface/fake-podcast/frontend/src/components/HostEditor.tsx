import type { HostConfig } from '../types';

interface Props {
  label: string;
  host: HostConfig;
  onChange: (h: HostConfig) => void;
}

export default function HostEditor({ label, host, onChange }: Props) {
  return (
    <div className="card space-y-2">
      <div className="flex items-center gap-2">
        <span
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: host.color }}
        />
        <span className="text-sm font-semibold text-gray-700">{label}</span>
      </div>
      <input
        className="input-field text-sm"
        placeholder="Host name"
        value={host.name}
        onChange={(e) => onChange({ ...host, name: e.target.value })}
      />
      <textarea
        className="input-field text-sm resize-none"
        rows={2}
        placeholder="Personality description..."
        value={host.personality}
        onChange={(e) => onChange({ ...host, personality: e.target.value })}
      />
    </div>
  );
}
