import React from 'react';

const ProgressBar = ({ progress, className = '', showPercent = true }) => {
    return (
        <div className={`w-full ${className}`}>
            <div className="w-full bg-slate-700 rounded-full h-2.5 overflow-hidden">
                <div
                    className="bg-gradient-to-r from-primary-600 to-primary-400 h-2.5 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                >
                    <div className="h-full w-full bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
                </div>
            </div>
            {showPercent && (
                <p className="text-xs text-slate-400 mt-1 text-right">
                    {Math.round(progress)}%
                </p>
            )}
        </div>
    );
};

export default ProgressBar;
