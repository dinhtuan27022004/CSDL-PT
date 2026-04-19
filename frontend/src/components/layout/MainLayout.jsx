import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';
import Toast from '../common/Toast';

const MainLayout = () => {
    return (
        <div className="min-h-screen flex flex-col bg-slate-950">
            <Header />

            <div className="flex flex-1 overflow-hidden">
                <Sidebar />

                <main className="flex-1 overflow-y-auto">
                    <div className="w-full px-8 py-6">
                        <Outlet />
                    </div>
                </main>
            </div>


            <Toast />
        </div>
    );
};

export default MainLayout;
