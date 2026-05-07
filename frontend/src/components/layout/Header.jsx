import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
    Database, 
    Upload, 
    Search, 
    GitCompare 
} from 'lucide-react';

const Header = () => {
    const navItems = [
        { to: '/import', icon: Upload, label: 'Import' },
        { to: '/search', icon: Search, label: 'Search' },
        { to: '/evaluate', icon: GitCompare, label: 'Evaluate' },
        { to: '/data', icon: Database, label: 'Data' },
    ];

    return (
        <header className="bg-slate-900/50 backdrop-blur-md border-b border-slate-700/50 px-8 py-3 sticky top-0 z-50">
            <div className="flex items-center justify-between gap-8">
                {/* Logo & Info */}
                <div className="flex items-center gap-3 min-w-fit">
                   
                </div>

                {/* Navigation Menu */}
                <nav className="flex items-center gap-1 bg-slate-950/40 p-1 rounded-xl border border-slate-800/50">
                    {navItems.map((item) => (
                        <NavLink
                            key={item.to}
                            to={item.to}
                            className={({ isActive }) =>
                                `flex items-center gap-2 px-6 py-2.5 rounded-lg transition-all duration-300 group ${isActive
                                    ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/20'
                                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                                }`
                            }
                        >
                            <item.icon className={`w-4 h-4 transition-transform duration-300 group-hover:scale-110`} />
                            <span className="text-xs font-black uppercase tracking-widest">{item.label}</span>
                        </NavLink>
                    ))}
                </nav>

                {/* Status Indicator */}
                <div className="flex items-center gap-4 min-w-fit">
                    
                </div>
            </div>
        </header>
    );
};

export default Header;
