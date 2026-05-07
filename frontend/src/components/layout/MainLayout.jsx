import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import Toast from '../common/Toast';

const MainLayout = () => {
    return (
        <div className="min-h-screen flex flex-col bg-slate-950">
            <Header />

            <main className="flex-1 overflow-y-auto">
                <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    <Outlet />
                </div>
            </main>

            <Toast />
        </div>
    );
};

export default MainLayout;
