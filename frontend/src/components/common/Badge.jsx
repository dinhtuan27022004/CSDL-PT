import React from 'react';

const Badge = ({ children, variant = 'default', className = '' }) => {
    const variants = {
        default: 'bg-slate-700 text-slate-200',
        success: 'bg-green-600/20 text-green-400 border border-green-500/30',
        warning: 'bg-yellow-600/20 text-yellow-400 border border-yellow-500/30',
        error: 'bg-red-600/20 text-red-400 border border-red-500/30',
        info: 'bg-blue-600/20 text-blue-400 border border-blue-500/30',
        primary: 'bg-primary-600/20 text-primary-400 border border-primary-500/30',
    };

    return (
        <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variants[variant]} ${className}`}
        >
            {children}
        </span>
    );
};

export default Badge;
