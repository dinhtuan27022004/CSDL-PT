import React from 'react';

const Button = ({
    children,
    variant = 'primary',
    size = 'md',
    disabled = false,
    onClick,
    className = '',
    type = 'button',
    ...props
}) => {
    const variants = {
        primary: 'bg-primary-600 hover:bg-primary-700 text-white disabled:bg-primary-800/50',
        secondary: 'bg-slate-700 hover:bg-slate-600 text-white disabled:bg-slate-800/50',
        outline: 'border-2 border-primary-600 text-primary-400 hover:bg-primary-600/10 disabled:border-slate-700 disabled:text-slate-600',
        ghost: 'text-slate-300 hover:bg-slate-800 disabled:text-slate-600',
        danger: 'bg-red-600 hover:bg-red-700 text-white disabled:bg-red-800/50',
    };

    const sizes = {
        sm: 'px-3 py-1.5 text-sm',
        md: 'px-4 py-2 text-base',
        lg: 'px-6 py-3 text-lg',
    };

    return (
        <button
            type={type}
            onClick={onClick}
            disabled={disabled}
            className={`
        inline-flex items-center justify-center
        rounded-lg font-medium
        transition-all duration-200
        disabled:cursor-not-allowed disabled:opacity-50
        ${variants[variant]}
        ${sizes[size]}
        ${className}
      `}
            {...props}
        >
            {children}
        </button>
    );
};

export default Button;
