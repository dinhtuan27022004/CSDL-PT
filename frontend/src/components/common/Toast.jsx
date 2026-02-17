import React, { useEffect } from 'react';
import useImageStore from '../../store/useImageStore';
import { CheckCircle, XCircle, AlertCircle, Info } from 'lucide-react';

const Toast = () => {
    const { toasts, removeToast } = useImageStore();

    const icons = {
        success: <CheckCircle className="w-5 h-5" />,
        error: <XCircle className="w-5 h-5" />,
        warning: <AlertCircle className="w-5 h-5" />,
        info: <Info className="w-5 h-5" />,
    };

    const styles = {
        success: 'bg-green-600/20 border-green-500/30 text-green-400',
        error: 'bg-red-600/20 border-red-500/30 text-red-400',
        warning: 'bg-yellow-600/20 border-yellow-500/30 text-yellow-400',
        info: 'bg-blue-600/20 border-blue-500/30 text-blue-400',
    };

    return (
        <div className="fixed bottom-4 right-4 z-50 space-y-2">
            {toasts.map((toast) => (
                <div
                    key={toast.id}
                    className={`
            flex items-center gap-3 min-w-[300px] max-w-md
            px-4 py-3 rounded-lg border
            backdrop-blur-sm
            animate-slide-in-right
            ${styles[toast.type]}
          `}
                >
                    {icons[toast.type]}
                    <p className="flex-1 text-sm font-medium">{toast.message}</p>
                    <button
                        onClick={() => removeToast(toast.id)}
                        className="hover:opacity-70 transition-opacity"
                    >
                        <XCircle className="w-4 h-4" />
                    </button>
                </div>
            ))}
        </div>
    );
};

export default Toast;
