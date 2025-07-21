import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

@dataclass
class SimulationParameters:
    """All tunable parameters for the restaurant simulation"""
    
    # ===== CUSTOMER HAPPINESS THRESHOLDS (TUNE HERE) =====
    walkin_happiness_threshold: float = 20.0  # Walk-in customers unhappy if wait > 20 min
    driveThru_happiness_threshold: float = 10.0  # Drive-thru customers unhappy if wait > 10 min
    
    # ===== DEMAND FUNCTION PARAMETERS =====
    # demand_type = base_demand + alpha * past_avg_happiness - beta * price + noise
    alpha: float = 0.8  # How much past happiness affects demand
    beta: float = 0.3   # How much price affects demand
    
    # Base demand rates per hour for each customer type
    base_walkin_demand: float = 20.0  # Walk-in customers per hour
    base_driveThru_demand: float = 30.0  # Drive-thru customers per hour
    
    demand_variability: float = 0.2  # Stochastic noise factor (0-1)
    
    # ===== STAFF PARAMETERS =====
    base_staff_count: int = 6 # Assuming 6 staff members available at baseline
    max_staff_count: int = 10  # Maximum staff available
    min_staff_count: int = 4  # Minimum staff available
    staff_cost_per_hour: float = 15.0
    service_time_per_customer: float = 5.0  # Minutes per customer per staff member
    
    # ===== PROMOTION PARAMETERS =====
    base_price: float = 10.0
    promotion_cost_per_hour: float = 100.0  # Fixed cost when promotion is active
    promotion_discount: float = 0.2  # 20% discount
    
    # ===== SIMULATION PARAMETERS =====
    happiness_memory_hours: int = 3  # How many hours of past happiness to consider
    simulation_hours: int = 24  # Total simulation time
    warmup_hours: int = 2  # Hours to ignore for statistics (warmup period)
    repetitions: int = 30  # Number of repetitions for each strategy to average results

class Customer:
    """Represents a customer with arrival time and type"""
    
    def __init__(self, customer_id: int, customer_type: str, arrival_time: float):
        self.id = customer_id
        self.type = customer_type  # 'walkin' or 'driveThru'
        self.arrival_time = arrival_time
        self.service_start_time = None
        self.service_end_time = None
        self.waiting_time = 0
        self.is_happy = False

class RestaurantSimulation:
    """Main simulation class using SimPy"""
    
    def __init__(self, params: SimulationParameters, strategy: str = 'baseline'):
        self.params = params
        self.strategy = strategy
        self.env = simpy.Environment()
        
        # Resources (staff)
        self.staff = None
        self.current_staff_count = 0
        
        # Data collection
        self.customers: List[Customer] = []
        self.hourly_stats = []
        self.happiness_history = []  # Track happiness over time
        
        # Current hour tracking
        self.current_hour = 0
        self.customers_this_hour = {'walkin': [], 'driveThru': []}
        self.current_hour += 1  # Start at hour 1
        
        # Strategy configuration
        self.strategy_config = self._get_strategy_config(strategy)
        
    def _get_strategy_config(self, strategy: str) -> Dict:
        """Get configuration for different strategies"""
        configs = {
            'baseline': {'staff_multiplier': 1.0, 'promotion': False},
            'reduce_staff': {'staff_multiplier': 0.7, 'promotion': False},  # 30% reduction
            'promotion': {'staff_multiplier': 1.0, 'promotion': True},
            'combined': {'staff_multiplier': 0.8, 'promotion': True}  # 20% reduction + promotion
        }
        return configs.get(strategy, configs['baseline'])
    
    def _calculate_demand_rate(self, customer_type: str, past_avg_happiness: float, current_price: float) -> float:
        """Calculate demand rate for specific customer type using the demand function"""
        
        if customer_type == 'walkin':
            base_demand = self.params.base_walkin_demand
        else:  # driveThru
            base_demand = self.params.base_driveThru_demand
        
        # Demand function: demand = base + alpha * happiness - beta * price
        demand_rate = (base_demand + 
                      self.params.alpha * past_avg_happiness - 
                      self.params.beta * current_price)
        
        # Add stochastic noise
        noise = np.random.normal(0, self.params.demand_variability * base_demand)
        demand_rate = max(0.1, demand_rate + noise)  # Ensure positive demand
        
        return demand_rate
    
    def _get_past_avg_happiness(self) -> float:
        """Calculate average happiness from recent history"""
        if len(self.happiness_history) == 0:
            return 0.5  # Neutral starting point
        
        # Get recent happiness scores
        recent_happiness = self.happiness_history[-self.params.happiness_memory_hours:]
        return np.mean(recent_happiness) if recent_happiness else 0.5
    
    def _get_current_price(self) -> float:
        """Get current price based on strategy"""
        if self.strategy_config['promotion']:
            return self.params.base_price * (1 - self.params.promotion_discount)
        return self.params.base_price
    
    def customer_generator(self, customer_type: str):
        """Generate customers of a specific type"""
        customer_id = 0
        
        while True:
            # Calculate current demand rate
            past_happiness = self._get_past_avg_happiness()
            current_price = self._get_current_price()
            demand_rate = self._calculate_demand_rate(customer_type, past_happiness, current_price)
            
            # Convert hourly rate to inter-arrival time (exponential distribution)
            inter_arrival_time = np.random.exponential(60 / demand_rate)  # Convert to minutes
            
            yield self.env.timeout(inter_arrival_time)
            
            # Create customer
            customer = Customer(customer_id, customer_type, self.env.now)
            customer_id += 1
            
            # Track which hour this customer belongs to
            hour = int(self.env.now // 60)
            if hour < self.params.simulation_hours:
                self.customers_this_hour[customer_type].append(customer)
            
            # Start service process
            self.env.process(self.serve_customer(customer))
    
    def serve_customer(self, customer: Customer):
        """Serve a customer"""
        # Request staff resource
        with self.staff.request() as request:
            yield request
            
            # Customer starts being served
            customer.service_start_time = self.env.now
            customer.waiting_time = customer.service_start_time - customer.arrival_time
            
            # Determine happiness based on waiting time
            threshold = (self.params.walkin_happiness_threshold if customer.type == 'walkin' 
                        else self.params.driveThru_happiness_threshold)
            customer.is_happy = customer.waiting_time <= threshold
            
            # Service time
            service_time = np.random.exponential(self.params.service_time_per_customer)
            yield self.env.timeout(service_time)
            
            customer.service_end_time = self.env.now
            self.customers.append(customer)
    
    def hourly_statistics(self):
        """Collect hourly statistics"""
        while True:
            yield self.env.timeout(60)  # Wait for one hour
            
            if self.current_hour >= self.params.simulation_hours:
                break
            
            # Calculate statistics for this hour
            walkin_customers = self.customers_this_hour['walkin']
            driveThru_customers = self.customers_this_hour['driveThru']
            all_customers = walkin_customers + driveThru_customers
            
            if all_customers:
                # Calculate happiness metrics
                happy_customers = sum(1 for c in all_customers if c.is_happy)
                total_customers = len(all_customers)
                happiness_rate = happy_customers / total_customers if total_customers > 0 else 0
                
                # Calculate waiting times
                avg_wait_walkin = np.mean([c.waiting_time for c in walkin_customers]) if walkin_customers else 0
                avg_wait_driveThru = np.mean([c.waiting_time for c in driveThru_customers]) if driveThru_customers else 0
                
                # Calculate financial metrics
                current_price = self._get_current_price()
                revenue = total_customers * current_price
                staff_cost = self.current_staff_count * self.params.staff_cost_per_hour
                promotion_cost = self.params.promotion_cost_per_hour if self.strategy_config['promotion'] else 0
                profit = revenue - staff_cost - promotion_cost
                
                # Store statistics
                hour_stats = {
                    'hour': self.current_hour,
                    'walkin_demand': len(walkin_customers),
                    'driveThru_demand': len(driveThru_customers),
                    'total_demand': total_customers,
                    'happiness_rate': happiness_rate,
                    'avg_wait_walkin': avg_wait_walkin,
                    'avg_wait_driveThru': avg_wait_driveThru,
                    'current_price': current_price,
                    'staff_count': self.current_staff_count,
                    'revenue': revenue,
                    'staff_cost': staff_cost,
                    'promotion_cost': promotion_cost,
                    'profit': profit,
                    'strategy': self.strategy
                }
                
                self.hourly_stats.append(hour_stats)
                self.happiness_history.append(happiness_rate)
                
                print(f"Hour {self.current_hour}: {total_customers} customers, "
                      f"{happiness_rate:.2%} happy, Profit: ${profit:.2f}")
            
            # Reset for next hour
            self.customers_this_hour = {'walkin': [], 'driveThru': []}
            self.current_hour += 1
            # Adjust staff count based on strategy
            if total_customers > 60:
                new_staff = self.params.max_staff_count
            elif total_customers < 20:
                new_staff = self.params.min_staff_count
            else:
                new_staff = self.params.base_staff_count
                
    
    def run_simulation(self):
        """Run the complete simulation"""
        print(f"Starting simulation with strategy: {self.strategy}")
        
        # Initialize staff resource
        self.current_staff_count = int(self.params.base_staff_count * self.strategy_config['staff_multiplier'])
        self.staff = simpy.Resource(self.env, capacity=self.current_staff_count)
        
        # Start customer generation processes
        self.env.process(self.customer_generator('walkin'))
        self.env.process(self.customer_generator('driveThru'))
        
        # Start statistics collection
        self.env.process(self.hourly_statistics())
        
        # Run simulation
        self.env.run(until=self.params.simulation_hours * 60)  # Convert hours to minutes
        
        print(f"Simulation completed. Total customers served: {len(self.customers)}")
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """Get simulation results"""
        if not self.hourly_stats:
            return {}
        
        # Filter out warmup period
        valid_stats = [stat for stat in self.hourly_stats if stat['hour'] >= self.params.warmup_hours]
        
        if not valid_stats:
            return {}
        
        # Calculate summary statistics
        total_profit = sum(stat['profit'] for stat in valid_stats)
        avg_happiness = np.mean([stat['happiness_rate'] for stat in valid_stats])
        avg_wait_walkin = np.mean([stat['avg_wait_walkin'] for stat in valid_stats])
        avg_wait_driveThru = np.mean([stat['avg_wait_driveThru'] for stat in valid_stats])
        total_customers = sum(stat['total_demand'] for stat in valid_stats)
        
        return {
            'strategy': self.strategy,
            'total_profit': total_profit,
            'avg_happiness': avg_happiness,
            'avg_wait_walkin': avg_wait_walkin,
            'avg_wait_driveThru': avg_wait_driveThru,
            'total_customers': total_customers,
            'hourly_stats': valid_stats
        }

def run_multiple_simulations(strategy: str, params: SimulationParameters, repetitions: int = None) -> Dict:
    """Run multiple simulations for a strategy and calculate averages"""
    if repetitions is None:
        repetitions = params.repetitions
    
    print(f"Running {repetitions} simulations for strategy: {strategy}")
    
    all_results = []
    all_hourly_stats = []
    
    for rep in range(repetitions):
        if (rep + 1) % 5 == 0:  # Progress indicator every 5 repetitions
            print(f"  Completed {rep + 1}/{repetitions} repetitions...")
        
        # Set different random seed for each repetition
        np.random.seed(42 + rep * 100)
        random.seed(42 + rep * 100)
        
        sim = RestaurantSimulation(params, strategy)
        result = sim.run_simulation()
        
        if result and result.get('hourly_stats'):
            all_results.append(result)
            all_hourly_stats.append(result['hourly_stats'])
    
    if not all_results:
        return {}
    
    # Calculate averaged results
    averaged_result = calculate_averaged_results(all_results, all_hourly_stats, strategy)
    
    print(f"  Completed all {repetitions} repetitions for {strategy}")
    return averaged_result

def calculate_averaged_results(all_results: List[Dict], all_hourly_stats: List[List[Dict]], strategy: str) -> Dict:
    """Calculate averaged results across multiple repetitions"""
    
    # Calculate summary statistics averages
    total_profits = [result['total_profit'] for result in all_results]
    avg_happiness_scores = [result['avg_happiness'] for result in all_results]
    avg_wait_walkin_scores = [result['avg_wait_walkin'] for result in all_results]
    avg_wait_driveThru_scores = [result['avg_wait_driveThru'] for result in all_results]
    total_customers_scores = [result['total_customers'] for result in all_results]
    
    # Calculate averages and standard deviations for summary stats
    avg_total_profit = np.mean(total_profits)
    std_total_profit = np.std(total_profits)
    avg_avg_happiness = np.mean(avg_happiness_scores)
    std_avg_happiness = np.std(avg_happiness_scores)
    avg_avg_wait_walkin = np.mean(avg_wait_walkin_scores)
    avg_avg_wait_driveThru = np.mean(avg_wait_driveThru_scores)
    avg_total_customers = np.mean(total_customers_scores)
    
    # Calculate hourly averages
    # First, organize data by hour
    hourly_data = defaultdict(list)
    
    for hourly_stats in all_hourly_stats:
        for hour_stat in hourly_stats:
            hour = hour_stat['hour']
            hourly_data[hour].append(hour_stat)
    
    # Calculate averages for each hour
    averaged_hourly_stats = []
    for hour in sorted(hourly_data.keys()):
        hour_data = hourly_data[hour]
        
        # Calculate averages for this hour
        avg_hour_stat = {
            'hour': hour,
            'walkin_demand': np.mean([h['walkin_demand'] for h in hour_data]),
            'walkin_demand_std': np.std([h['walkin_demand'] for h in hour_data]),
            'driveThru_demand': np.mean([h['driveThru_demand'] for h in hour_data]),
            'driveThru_demand_std': np.std([h['driveThru_demand'] for h in hour_data]),
            'total_demand': np.mean([h['total_demand'] for h in hour_data]),
            'total_demand_std': np.std([h['total_demand'] for h in hour_data]),
            'happiness_rate': np.mean([h['happiness_rate'] for h in hour_data]),
            'happiness_rate_std': np.std([h['happiness_rate'] for h in hour_data]),
            'avg_wait_walkin': np.mean([h['avg_wait_walkin'] for h in hour_data]),
            'avg_wait_walkin_std': np.std([h['avg_wait_walkin'] for h in hour_data]),
            'avg_wait_driveThru': np.mean([h['avg_wait_driveThru'] for h in hour_data]),
            'avg_wait_driveThru_std': np.std([h['avg_wait_driveThru'] for h in hour_data]),
            'current_price': np.mean([h['current_price'] for h in hour_data]),
            'staff_count': np.mean([h['staff_count'] for h in hour_data]),
            'revenue': np.mean([h['revenue'] for h in hour_data]),
            'revenue_std': np.std([h['revenue'] for h in hour_data]),
            'staff_cost': np.mean([h['staff_cost'] for h in hour_data]),
            'promotion_cost': np.mean([h['promotion_cost'] for h in hour_data]),
            'profit': np.mean([h['profit'] for h in hour_data]),
            'profit_std': np.std([h['profit'] for h in hour_data]),
            'strategy': strategy,
            'repetitions': len(hour_data)
        }
        averaged_hourly_stats.append(avg_hour_stat)
    
    return {
        'strategy': strategy,
        'total_profit': avg_total_profit,
        'total_profit_std': std_total_profit,
        'avg_happiness': avg_avg_happiness,
        'avg_happiness_std': std_avg_happiness,
        'avg_wait_walkin': avg_avg_wait_walkin,
        'avg_wait_driveThru': avg_avg_wait_driveThru,
        'total_customers': avg_total_customers,
        'hourly_stats': averaged_hourly_stats,
        'repetitions': len(all_results),
        'raw_results': all_results  # Keep raw results for detailed analysis if needed
    }

def run_single_simulation(strategy: str = 'baseline', params: SimulationParameters = None) -> Dict:
    """Run a single simulation with given strategy"""
    if params is None:
        params = SimulationParameters()
    
    sim = RestaurantSimulation(params, strategy)
    results = sim.run_simulation()
    return results

def compare_strategies(params: SimulationParameters = None) -> Dict[str, Dict]:
    """Compare all strategies using multiple repetitions"""
    if params is None:
        params = SimulationParameters()
    
    strategies = ['baseline', 'reduce_staff', 'promotion', 'combined']
    results = {}
    
    print(f"Running strategy comparison with {params.repetitions} repetitions each...")
    print("This may take a few minutes...")
    
    for strategy in strategies:
        print(f"\n--- Running {strategy} strategy ({params.repetitions} repetitions) ---")
        results[strategy] = run_multiple_simulations(strategy, params)
    
    return results

def plot_results(results: Dict[str, Dict]):
    """Plot comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    strategies = list(results.keys())
    
    # Extract data for plotting
    profits = [results[s]['total_profit'] for s in strategies]
    happiness = [results[s]['avg_happiness'] * 100 for s in strategies]  # Convert to percentage
    wait_walkin = [results[s]['avg_wait_walkin'] for s in strategies]
    wait_driveThru = [results[s]['avg_wait_driveThru'] for s in strategies]
    
    # Plot 1: Total Profit
    axes[0, 0].bar(strategies, profits, color=['blue', 'red', 'green', 'purple'])
    axes[0, 0].set_title('Total Profit by Strategy')
    axes[0, 0].set_ylabel('Profit ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Average Happiness
    axes[0, 1].bar(strategies, happiness, color=['blue', 'red', 'green', 'purple'])
    axes[0, 1].set_title('Average Customer Happiness by Strategy')
    axes[0, 1].set_ylabel('Happiness (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Average Waiting Times
    x = np.arange(len(strategies))
    width = 0.35
    axes[1, 0].bar(x - width/2, wait_walkin, width, label='Walk-in', color='lightblue')
    axes[1, 0].bar(x + width/2, wait_driveThru, width, label='Drive-thru', color='lightcoral')
    axes[1, 0].set_title('Average Waiting Time by Strategy')
    axes[1, 0].set_ylabel('Waiting Time (minutes)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(strategies, rotation=45)
    axes[1, 0].legend()
    
    # Plot 4: Hourly profit trends (for baseline strategy)
    if 'baseline' in results and 'hourly_stats' in results['baseline']:
        baseline_stats = results['baseline']['hourly_stats']
        hours = [stat['hour'] for stat in baseline_stats]
        hourly_profits = [stat['profit'] for stat in baseline_stats]
        axes[1, 1].plot(hours, hourly_profits, marker='o', color='blue')
        axes[1, 1].set_title('Hourly Profit Trend (Baseline)')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Profit ($)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_time_series_analysis(results: Dict[str, Dict]):
    """Plot cumulative metrics over time for all strategies"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'baseline': 'blue', 'reduce_staff': 'red', 'promotion': 'green', 'combined': 'purple'}
    
    for strategy, result in results.items():
        if not result or 'hourly_stats' not in result:
            continue
            
        stats = result['hourly_stats']
        hours = [stat['hour'] for stat in stats]
        
        # Calculate cumulative metrics
        hourly_profits = [stat['profit'] for stat in stats]
        cumulative_profits = np.cumsum(hourly_profits)
        
        happiness_rates = [stat['happiness_rate'] * 100 for stat in stats]  # Convert to percentage
        
        hourly_customers = [stat['total_demand'] for stat in stats]
        cumulative_customers = np.cumsum(hourly_customers)
        
        color = colors.get(strategy, 'black')
        
        # Plot 1: Cumulative Profit Over Time
        axes[0, 0].plot(hours, cumulative_profits, marker='o', color=color, 
                       label=strategy.title().replace('_', ' '), linewidth=2, markersize=4)
        
        # Plot 2: Happiness Percentage Over Time
        axes[0, 1].plot(hours, happiness_rates, marker='s', color=color,
                       label=strategy.title().replace('_', ' '), linewidth=2, markersize=4)
        
        # Plot 3: Cumulative Number of Customers Over Time
        axes[1, 0].plot(hours, cumulative_customers, marker='^', color=color,
                       label=strategy.title().replace('_', ' '), linewidth=2, markersize=4)
        
        # Plot 4: Hourly Profit Comparison (All Strategies)
        axes[1, 1].plot(hours, hourly_profits, marker='d', color=color,
                       label=strategy.title().replace('_', ' '), linewidth=2, markersize=4)
    
    # Customize Plot 1: Cumulative Profit
    axes[0, 0].set_title('Cumulative Profit Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Cumulative Profit ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Customize Plot 2: Happiness Percentage
    axes[0, 1].set_title('Customer Happiness Percentage Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Happiness (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='70% Target')
    
    # Customize Plot 3: Cumulative Customers
    axes[1, 0].set_title('Cumulative Number of Customers Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Cumulative Customers')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Customize Plot 4: Hourly Profit Comparison
    axes[1, 1].set_title('Hourly Profit Comparison (All Strategies)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Hourly Profit ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_detailed_strategy_analysis(results: Dict[str, Dict]):
    """Plot detailed analysis for each strategy separately"""
    strategies = list(results.keys())
    n_strategies = len(strategies)
    
    fig, axes = plt.subplots(n_strategies, 3, figsize=(18, 5*n_strategies))
    
    # Handle case where there's only one strategy
    if n_strategies == 1:
        axes = axes.reshape(1, -1)
    
    colors = {'baseline': 'blue', 'reduce_staff': 'red', 'promotion': 'green', 'combined': 'purple'}
    
    for i, (strategy, result) in enumerate(results.items()):
        if not result or 'hourly_stats' not in result:
            continue
            
        stats = result['hourly_stats']
        hours = [stat['hour'] for stat in stats]
        
        # Data extraction
        hourly_profits = [stat['profit'] for stat in stats]
        cumulative_profits = np.cumsum(hourly_profits)
        happiness_rates = [stat['happiness_rate'] * 100 for stat in stats]
        walkin_demands = [stat['walkin_demand'] for stat in stats]
        driveThru_demands = [stat['driveThru_demand'] for stat in stats]
        total_demands = [stat['total_demand'] for stat in stats]
        cumulative_customers = np.cumsum(total_demands)
        
        color = colors.get(strategy, 'black')
        
        # Plot 1: Cumulative Profit for this strategy
        axes[i, 0].plot(hours, cumulative_profits, color=color, linewidth=3, marker='o', markersize=6)
        axes[i, 0].set_title(f'{strategy.title().replace("_", " ")} - Cumulative Profit', 
                            fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('Hour')
        axes[i, 0].set_ylabel('Cumulative Profit ($)')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Add profit trend line
        z = np.polyfit(hours, cumulative_profits, 1)
        p = np.poly1d(z)
        axes[i, 0].plot(hours, p(hours), "--", color='red', alpha=0.8, 
                       label=f'Trend: ${z[0]:.1f}/hour')
        axes[i, 0].legend()
        
        # Plot 2: Customer Happiness and Demand
        ax2_twin = axes[i, 1].twinx()
        
        # Happiness on left axis
        line1 = axes[i, 1].plot(hours, happiness_rates, color='green', linewidth=3, 
                               marker='s', markersize=6, label='Happiness %')
        axes[i, 1].set_ylabel('Happiness (%)', color='green')
        axes[i, 1].tick_params(axis='y', labelcolor='green')
        
        # Total demand on right axis
        line2 = ax2_twin.plot(hours, total_demands, color='orange', linewidth=3, 
                             marker='^', markersize=6, label='Total Demand')
        ax2_twin.set_ylabel('Total Customers/Hour', color='orange')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        
        axes[i, 1].set_title(f'{strategy.title().replace("_", " ")} - Happiness vs Demand', 
                            fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel('Hour')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[i, 1].legend(lines, labels, loc='upper left')
        
        # Plot 3: Customer Type Breakdown and Cumulative Count
        ax3_twin = axes[i, 2].twinx()
        
        # Stacked bar chart for customer types on left axis
        axes[i, 2].bar(hours, walkin_demands, color='lightblue', alpha=0.7, label='Walk-in')
        axes[i, 2].bar(hours, driveThru_demands, bottom=walkin_demands, 
                      color='lightcoral', alpha=0.7, label='Drive-thru')
        axes[i, 2].set_ylabel('Customers/Hour')
        
        # Cumulative customers on right axis
        ax3_twin.plot(hours, cumulative_customers, color='black', linewidth=3, 
                     marker='d', markersize=6, label='Cumulative Total')
        ax3_twin.set_ylabel('Cumulative Customers', color='black')
        ax3_twin.tick_params(axis='y', labelcolor='black')
        
        axes[i, 2].set_title(f'{strategy.title().replace("_", " ")} - Customer Analysis', 
                            fontsize=12, fontweight='bold')
        axes[i, 2].set_xlabel('Hour')
        axes[i, 2].legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create simulation parameters
    params = SimulationParameters()
    
    print("Fast Food Restaurant Simulation using SimPy")
    print("=" * 50)
    
    # Run single simulation
    print("\nRunning single simulation (baseline strategy)...")
    single_result = run_single_simulation('baseline', params)
    
    if single_result:
        print(f"\nSingle Simulation Results:")
        print(f"Strategy: {single_result['strategy']}")
        print(f"Total Profit: ${single_result['total_profit']:.2f}")
        print(f"Average Happiness: {single_result['avg_happiness']:.2%}")
        print(f"Average Wait Time (Walk-in): {single_result['avg_wait_walkin']:.2f} minutes")
        print(f"Average Wait Time (Drive-thru): {single_result['avg_wait_driveThru']:.2f} minutes")
        print(f"Total Customers: {single_result['total_customers']}")
    
    # Run strategy comparison
    print("\n" + "=" * 50)
    comparison_results = compare_strategies(params)
    
    # Print comparison summary
    print("\nStrategy Comparison Summary:")
    print("-" * 80)
    print(f"{'Strategy':<15} {'Profit':<10} {'Happiness':<12} {'Walk-in Wait':<13} {'Drive-thru Wait':<15}")
    print("-" * 80)
    
    for strategy, result in comparison_results.items():
        if result:  # Check if result is not empty
            print(f"{strategy:<15} ${result['total_profit']:<9.2f} {result['avg_happiness']:<11.2%} "
                  f"{result['avg_wait_walkin']:<12.2f} {result['avg_wait_driveThru']:<14.2f}")
    
    # Plot results
    if comparison_results:
        print("\nGenerating strategy comparison charts...")
        plot_results(comparison_results)
        
        print("Generating time series analysis...")
        plot_time_series_analysis(comparison_results)
        
        print("Generating detailed strategy analysis...")
        plot_detailed_strategy_analysis(comparison_results)
    
    # Create detailed DataFrame for analysis
    if comparison_results:
        all_hourly_data = []
        for strategy, result in comparison_results.items():
            if result and 'hourly_stats' in result:
                for stat in result['hourly_stats']:
                    all_hourly_data.append(stat)
        
        if all_hourly_data:
            df = pd.DataFrame(all_hourly_data)
            
            # Show sample of detailed data
            print("\nSample of Detailed Hourly Data:")
            print(df.head(10).to_string(index=False))
            
            # Save to CSV for further analysis
            df.to_csv('restaurant_simulation_results.csv', index=False)
            print("\nDetailed results saved to 'restaurant_simulation_results.csv'")

if __name__ == "__main__":
    main()