import React from 'react';

const Card = ({ children, className = '', hover = false, onClick }) => {
    return (
        <div
            onClick={onClick}
            className={`
        bg-slate-800/50 backdrop-blur-sm
        border border-slate-700/50
        rounded-lg
        transition-all duration-200
        ${hover ? 'hover:border-primary-500/50 hover:shadow-lg hover:shadow-primary-500/10 cursor-pointer' : ''}
        ${className}
      `}
        >
            {children}
        </div>
    );
};

export default Card;
