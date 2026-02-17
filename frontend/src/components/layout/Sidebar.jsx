import React from 'react';
import { NavLink } from 'react-router-dom';
import {
    Grid3x3,
    GitCompare,
    Upload,
    Settings,
    Search
} from 'lucide-react';

const Sidebar = () => {
    const navItems = [
        { to: '/import', icon: Upload, label: 'Import' },
        { to: '/search', icon: Search, label: 'Search' },
    ];

    return (
        <aside className="w-64 bg-slate-900/30 backdrop-blur-sm border-r border-slate-700/50 flex flex-col">
            <nav className="flex-1 px-4 py-6 space-y-2">
                {navItems.map((item) => (
                    <NavLink
                        key={item.to}
                        to={item.to}
                        className={({ isActive }) =>
                            `flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${isActive
                                ? 'bg-primary-600/20 text-primary-400 border border-primary-500/30'
                                : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                            }`
                        }
                    >
                        <item.icon className="w-5 h-5" />
                        <span className="font-medium">{item.label}</span>
                    </NavLink>
                ))}
            </nav>
        </aside>
    );
};

export default Sidebar;
