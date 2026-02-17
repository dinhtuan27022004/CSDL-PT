import React from 'react';
import { Database } from 'lucide-react';

const Header = () => {
    return (
        <header className="bg-slate-900/50 backdrop-blur-sm border-b border-slate-700/50 px-6 py-4">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="bg-gradient-to-br from-primary-500 to-primary-700 p-2 rounded-lg">
                        <Database className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-white">
                            Image Similarity Search
                        </h1>
                        <p className="text-xs text-slate-400">
                            Demo System - Pipeline Visualization
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-green-600/20 border border-green-500/30 rounded-lg">
                        <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                        <span className="text-sm text-green-400 font-medium">
                            DB: Connected (Mock)
                        </span>
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;
