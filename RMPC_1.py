import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Circle, Polygon
import time

class DifferentialDriveRobot:
    def __init__(self, b=0.5, dt=0.1):
        self.b = b
        self.dt = dt
        self.state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.history = []
    
    def dynamics(self, state, u):
        x, y, theta = state
        u1, u2 = u
        
        x_dot = (u1 + u2) / 2 * np.cos(theta)
        y_dot = (u1 + u2) / 2 * np.sin(theta)
        theta_dot = (u1 - u2) / self.b
        
        return np.array([x_dot, y_dot, theta_dot])
    
    def step(self, u, dt=None):
        if dt is None:
            dt = self.dt
        
        # Используем метод Рунге-Кутта 4-го порядка для большей точности
        k1 = self.dynamics(self.state, u)
        k2 = self.dynamics(self.state + 0.5 * dt * k1, u)
        k3 = self.dynamics(self.state + 0.5 * dt * k2, u)
        k4 = self.dynamics(self.state + dt * k3, u)
        
        self.state += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        self.history.append(self.state.copy())
        return self.state.copy()

class ImprovedNonlinearMPC:
    def __init__(self, N=15, dt=0.1, b=0.5):
        self.N = N
        self.dt = dt
        self.b = b
        
        self.Q = np.diag([15, 15, 8])
        self.R = np.diag([0.05, 0.05])
        self.P = np.diag([25, 25, 12])
        
        self.u_min = np.array([-2.0, -2.0])
        self.u_max = np.array([2.0, 2.0])
        
        self.obstacles = []
        self.previous_solution = None
        self.optimization_history = []
    
    def add_obstacle(self, x, y, radius):
        self.obstacles.append({'x': x, 'y': y, 'radius': radius})
    
    def obstacle_cost(self, x, y):
        cost = 0
        for obs in self.obstacles:
            dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
            safety_dist = obs['radius'] + 0.4
            
            if dist < safety_dist:
                cost += 500 * np.exp(-3 * (dist - obs['radius']))
            
            if dist < safety_dist + 0.5:
                cost += 50 / (dist - obs['radius'] + 0.1)
                
        return cost
    
    def discretized_dynamics(self, state, u):
        x, y, theta = state
        u1, u2 = u
        
        x_next = x + self.dt * (u1 + u2) / 2 * np.cos(theta)
        y_next = y + self.dt * (u1 + u2) / 2 * np.sin(theta)
        theta_next = theta + self.dt * (u1 - u2) / self.b
        
        return np.array([x_next, y_next, theta_next])
    
    def predict_trajectory(self, x0, U):
        """Прогнозирует траекторию на основе начального состояния и управлений"""
        trajectory = [x0.copy()]
        x = x0.copy()
        
        for k in range(U.shape[0]):
            x = self.discretized_dynamics(x, U[k])
            trajectory.append(x.copy())
            
        return np.array(trajectory)
    
    def cost_function(self, u_flat, x0, x_ref, u_prev=None):
        U = u_flat.reshape((self.N, 2))
        x = x0.copy()
        cost = 0
        
        for k in range(self.N):
            state_error = x - x_ref
            cost += state_error.T @ self.Q @ state_error
            cost += U[k].T @ self.R @ U[k]
            
            if u_prev is not None and k == 0:
                du = U[k] - u_prev
                cost += 0.1 * du.T @ du
            elif k > 0:
                du = U[k] - U[k-1]
                cost += 0.1 * du.T @ du
            
            cost += self.obstacle_cost(x[0], x[1])
            x = self.discretized_dynamics(x, U[k])
        
        state_error = x - x_ref
        cost += state_error.T @ self.P @ state_error
        
        angle_error = abs(x[2] - x_ref[2])
        if angle_error > np.pi:
            angle_error = 2*np.pi - angle_error
        cost += 5 * angle_error
        
        return cost
    
    def solve(self, x0, x_ref):
        if self.previous_solution is not None:
            u0 = np.roll(self.previous_solution, -2)
            u0[-2:] = u0[-4:-2]
        else:
            u0 = np.zeros(2 * self.N)
        
        bounds = []
        for i in range(self.N):
            bounds.extend([(self.u_min[0], self.u_max[0]), 
                          (self.u_min[1], self.u_max[1])])
        
        u_prev = None
        if self.previous_solution is not None:
            u_prev = self.previous_solution[:2]
        
        # Записываем начальное состояние оптимизации
        opt_info = {
            'step': len(self.optimization_history),
            'initial_guess': u0.copy(),
            'initial_cost': self.cost_function(u0, x0, x_ref, u_prev),
            'iterations': []
        }
        
        # Функция обратного вызова для отслеживания итераций
        def callback(xk):
            current_cost = self.cost_function(xk, x0, x_ref, u_prev)
            opt_info['iterations'].append({
                'iteration': len(opt_info['iterations']),
                'cost': current_cost,
                'solution': xk.copy()
            })
        
        result = minimize(
            self.cost_function, 
            u0, 
            args=(x0, x_ref, u_prev),
            method='SLSQP',
            bounds=bounds,
            callback=callback,
            options={'maxiter': 100, 'ftol': 1e-5}
        )
        
        opt_info['final_cost'] = result.fun
        opt_info['success'] = result.success
        opt_info['message'] = result.message
        opt_info['final_solution'] = result.x.copy() if result.success else None
        
        self.optimization_history.append(opt_info)
        
        if result.success:
            U_opt = result.x.reshape((self.N, 2))
            self.previous_solution = result.x
            return U_opt[0], self.predict_trajectory(x0, U_opt)
        else:
            print(f"Оптимизация не удалась: {result.message}")
            if self.previous_solution is not None:
                U_opt = self.previous_solution.reshape((self.N, 2))
                return U_opt[0], self.predict_trajectory(x0, U_opt)
            else:
                return np.array([0.0, 0.0]), np.array([x0])

def create_simpler_obstacle_course():
    obstacles = []
    
    key_obstacles = [
        (2.0, 2.0, 0.5),
        (4.0, 1.0, 0.4),
        (4.0, 3.5, 0.4),
        (6.0, 2.5, 0.6),
        (8.0, 1.5, 0.3),
        (8.0, 4.0, 0.3),
    ]
    obstacles.extend(key_obstacles)
    
    return obstacles

def real_time_mpc_simulation():
    """Функция симуляции MPC в реальном времени с визуализацией каждого шага"""
    
    # Инициализация
    robot = DifferentialDriveRobot(b=0.5, dt=0.1)
    mpc = ImprovedNonlinearMPC(N=15, dt=0.1, b=0.5)
    
    # Создаем упрощенную трассу с препятствиями
    obstacles = create_simpler_obstacle_course()
    for obs in obstacles:
        mpc.add_obstacle(obs[0], obs[1], obs[2])
    
    # Целевое состояние
    x_ref = np.array([10.0, 5.0, 0.0])
    
    # Настройка графики
    plt.ion()  # Включаем интерактивный режим
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Основной график - траектория
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 7)
    ax1.set_xlabel('X позиция')
    ax1.set_ylabel('Y позиция')
    ax1.set_title('Real-time MPC: Управление мобильным роботом')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # График управления
    ax2.set_xlabel('Шаг времени')
    ax2.set_ylabel('Управление')
    ax2.set_title('Управляющие воздействия')
    ax2.grid(True, alpha=0.3)
    
    # График стоимости оптимизации
    ax3.set_xlabel('Итерация оптимизации')
    ax3.set_ylabel('Стоимость')
    ax3.set_title('Сходимость MPC оптимизации')
    ax3.grid(True, alpha=0.3)
    
    # График состояния
    ax4.set_xlabel('Шаг времени')
    ax4.set_ylabel('Состояние')
    ax4.set_title('Эволюция состояний')
    ax4.grid(True, alpha=0.3)
    
    # Отрисовка препятствий
    for obs in obstacles:
        circle = Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.6)
        ax1.add_patch(circle)
    
    # Отрисовка цели
    ax1.plot(x_ref[0], x_ref[1], 'g*', markersize=20, label='Цель')
    
    # Отрисовка старта   
    ax1.plot(0, 0, 'go', markersize=10, label='Старт')
    
    # Инициализация элементов графики
    trajectory_line, = ax1.plot([], [], 'b-', linewidth=2, label='Траектория')
    predicted_trajectory, = ax1.plot([], [], 'm--', linewidth=1, alpha=0.7, label='Прогноз MPC')
    
    # Создаем робота как полигон
    robot_patch = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, 
                         facecolor='blue', alpha=0.7, edgecolor='black')
    ax1.add_patch(robot_patch)
    
    # Линии для управления
    u1_line, = ax2.plot([], [], 'b-', linewidth=2, label='u1 (правое колесо)')
    u2_line, = ax2.plot([], [], 'r-', linewidth=2, label='u2 (левое колесо)')
    ax2.legend()
    
    # Линия для стоимости оптимизации
    cost_line, = ax3.plot([], [], 'g-', linewidth=2, label='Текущая оптимизация')
    ax3.legend()
    
    # Линии для состояния
    x_line, = ax4.plot([], [], 'b-', linewidth=2, label='x')
    y_line, = ax4.plot([], [], 'r-', linewidth=2, label='y')
    theta_line, = ax4.plot([], [], 'g-', linewidth=2, label='θ')
    ax4.legend()
    
    # Текст с информацией
    info_text = ax1.text(0.5, 1.05, '', transform=ax1.transAxes, 
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Легенда
    ax1.legend(loc='upper left')
    
    # Хранилище данных
    states_history = [robot.state.copy()]
    controls_history = []
    predicted_trajectories = []
    
    max_steps = 200
    
    print("Запуск реального времени MPC...")
    print("Каждый шаг будет отображаться в реальном времени")
    print("Синий прямоугольник - робот")
    print("Красные круги - препятствия")
    print("Зеленая звезда - цель")
    print("Синяя линия - пройденная траектория")
    print("Пурпурная пунктирная линия - прогноз MPC")
    
    for step in range(max_steps):
        # Получаем текущее состояние
        current_state = robot.state.copy()
        
        # Решаем задачу MPC
        start_time = time.time()
        u_opt, predicted_traj = mpc.solve(current_state, x_ref)
        solve_time = time.time() - start_time
        
        # Применяем управление
        robot.step(u_opt)
        current_state = robot.state.copy()
        
        # Сохраняем данные
        states_history.append(current_state.copy())
        controls_history.append(u_opt.copy())
        predicted_trajectories.append(predicted_traj.copy())
        
        # Обновляем графики
        
        # 1. Траектория движения
        traj_x = [s[0] for s in states_history]
        traj_y = [s[1] for s in states_history]
        trajectory_line.set_data(traj_x, traj_y)
        
        # 2. Позиция робота
        x, y, theta = current_state
        length = 0.3
        width = 0.2
        vertices = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2]
        ])
        
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        rotated_vertices = vertices @ rot.T
        rotated_vertices[:, 0] += x
        rotated_vertices[:, 1] += y
        
        robot_patch.set_xy(rotated_vertices)
        
        # 3. Прогнозируемая траектория
        if len(predicted_traj) > 0:
            pred_x = [s[0] for s in predicted_traj]
            pred_y = [s[1] for s in predicted_traj]
            predicted_trajectory.set_data(pred_x, pred_y)
        
        # 4. Управляющие воздействия
        if step > 0:
            steps = list(range(step+1))
            u1_line.set_data(steps, [c[0] for c in controls_history])
            u2_line.set_data(steps, [c[1] for c in controls_history])
            ax2.set_xlim(0, max(step+1, 10))
            ax2.set_ylim(-2.5, 2.5)
        
        # 5. Стоимость оптимизации (последняя оптимизация)
        if mpc.optimization_history:
            last_opt = mpc.optimization_history[-1]
            if last_opt['iterations']:
                iterations = [it['iteration'] for it in last_opt['iterations']]
                costs = [it['cost'] for it in last_opt['iterations']]
                cost_line.set_data(iterations, costs)
                ax3.set_xlim(0, max(iterations) if iterations else 10)
                ax3.set_ylim(0, max(costs) * 1.1 if costs else 10)
        
        # 6. Состояния
        if step > 0:
            steps = list(range(step+2))  # +1 для начального состояния
            x_line.set_data(steps, [s[0] for s in states_history])
            y_line.set_data(steps, [s[1] for s in states_history])
            theta_line.set_data(steps, [s[2] for s in states_history])
            ax4.set_xlim(0, max(steps))
        
        # 7. Информационный текст
        dist_to_target = np.linalg.norm(current_state[:2] - x_ref[:2])
        info_text.set_text(f'Шаг: {step} | Позиция: ({x:.2f}, {y:.2f}) | Угол: {np.degrees(theta):.1f}° | До цели: {dist_to_target:.2f} | Время MPC: {solve_time:.3f}с')
        
        # Обновляем график
        plt.pause(0.1)
        
        # Проверяем достижение цели
        pos_error = np.linalg.norm(current_state[:2] - x_ref[:2])
        angle_error = abs(current_state[2] - x_ref[2])
        if angle_error > np.pi:
            angle_error = 2*np.pi - angle_error
            
        if pos_error < 0.4 and angle_error < 0.3:
            print(f"Цель достигнута на шаге {step}")
            break
            
        # Защита от застревания
        if step > 30 and np.linalg.norm(current_state[:2] - states_history[step-30][:2]) < 0.2:
            print(f"Робот застрял на шаге {step}, пробуем выйти...")
            u_random = np.random.uniform(-1, 1, 2)
            robot.step(u_random)
            states_history.append(robot.state.copy())
            controls_history.append(u_random)
    
    # Отключаем интерактивный режим
    plt.ioff()
    
    # Показываем финальный график
    plt.show()
    
    return states_history, controls_history, predicted_trajectories

if __name__ == "__main__":
    print("Real-time MPC для робота с дифференциальным приводом")
    print("Каждый шаг отображается в реальном времени")
    
    # Запускаем симуляцию в реальном времени
    states, controls, predictions = real_time_mpc_simulation()
    
    print("Симуляция завершена!")