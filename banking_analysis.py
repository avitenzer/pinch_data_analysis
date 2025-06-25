#!/usr/bin/env python3
"""
Banking Transactions Analysis Script
Calculates KPIs from cleaned_banking_transactions.csv:
1. Total monthly spending per category
2. Top 3 merchants by total spend
3. Average transaction size per week
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """Load and clean the banking transactions data"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Convert settlement_date to datetime (changed from transaction_date)
    df['settlement_date'] = pd.to_datetime(df['settlement_date'], errors='coerce')
    
    # Convert amount to numeric, handling various formats
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Remove only transactions where data quality is not clean
    initial_rows = len(df)
    df = df[df['data_quality_flags'] == 'clean']
    print(f"Removed {initial_rows - len(df)} rows with data quality issues (keeping only 'clean' transactions)")
    
    # Handle missing merchant names and categories (simple fallback)
    df['merchant_name'] = df['merchant_name'].fillna('Unknown Merchant')
    df['merchant_category'] = df['merchant_category'].fillna('Uncategorized')
    
    # Add currency conversion to USD
    print("Converting amounts to USD...")
    df = convert_to_usd(df)
    
    # Show merchant name distribution
    merchant_counts = df['merchant_name'].value_counts().head(10)
    print(f"\nTop 10 merchants:")
    for merchant, count in merchant_counts.items():
        print(f"  {merchant}: {count} transactions")
    
    print(f"\nLoaded {len(df)} transactions from {df['settlement_date'].min()} to {df['settlement_date'].max()}")
    return df

def convert_to_usd(df):
    """Convert all amounts to USD using specified exchange rates"""
    # Define exchange rates
    exchange_rates = {
        'USD': 1.0,
        'EUR': 1.1637,  # EURUSD = 1.1637
        'GBP': 1.3623,  # GBPUSD = 1.3623
        'CHF': 1.2421,  # CHFUSD = 1.2421
        'JPY': 1/145.41  # USDJPY = 145.41, so 1 JPY = 1/145.41 USD
    }
    
    # Initialize amount_usd column
    df['amount_usd'] = 0.0
    
    # Convert amounts based on currency
    for currency, rate in exchange_rates.items():
        mask = df['amount_ccy'] == currency
        df.loc[mask, 'amount_usd'] = df.loc[mask, 'amount'] * rate
        
        if mask.sum() > 0:
            print(f"  Converted {mask.sum()} {currency} transactions using rate {rate}")
    
    # Check for any unconverted currencies
    unconverted = df[~df['amount_ccy'].isin(exchange_rates.keys())]
    if len(unconverted) > 0:
        print(f"Warning: {len(unconverted)} transactions with unsupported currencies:")
        print(unconverted['amount_ccy'].value_counts())
        # For unsupported currencies, use original amount as fallback
        df.loc[~df['amount_ccy'].isin(exchange_rates.keys()), 'amount_usd'] = df.loc[~df['amount_ccy'].isin(exchange_rates.keys()), 'amount']
    
    print(f"Currency conversion complete. Total transactions: {len(df)}")
    return df

def calculate_monthly_spending_by_category(df):
    """Calculate total monthly spending per category"""
    print("\nCalculating monthly spending by category...")
    
    # Create year-month column using settlement_date
    df['year_month'] = df['settlement_date'].dt.to_period('M')
    
    # Focus on spending (negative amounts represent outgoing transactions)
    # We'll use absolute values for spending analysis, now using amount_usd
    spending_df = df[df['amount_usd'] < 0].copy()
    spending_df['spending_amount'] = abs(spending_df['amount_usd'])
    
    monthly_category_spending = spending_df.groupby(['year_month', 'merchant_category'])['spending_amount'].sum().reset_index()
    
    print(f"Analyzed spending across {spending_df['merchant_category'].nunique()} categories")
    print(f"Date range: {spending_df['year_month'].min()} to {spending_df['year_month'].max()}")
    
    return monthly_category_spending, spending_df

def get_top_merchants(df):
    """Get top 3 merchants by total spend"""
    print("\nCalculating top merchants by spend...")
    
    # Focus on spending (negative amounts), now using amount_usd
    spending_df = df[df['amount_usd'] < 0].copy()
    spending_df['spending_amount'] = abs(spending_df['amount_usd'])
    
    # Group by merchant and sum spending
    merchant_spending = spending_df.groupby('merchant_name')['spending_amount'].sum().sort_values(ascending=False)
    
    top_merchants = merchant_spending.head(3)
    
    print("Top 3 merchants by total spend:")
    for i, (merchant, amount) in enumerate(top_merchants.items(), 1):
        print(f"{i}. {merchant}: ${amount:,.2f}")
    
    return top_merchants, merchant_spending

def calculate_weekly_avg_transaction_size(df):
    """Calculate average transaction size per week"""
    print("\nCalculating weekly average transaction size...")
    
    # Create week column using settlement_date
    df['year_week'] = df['settlement_date'].dt.to_period('W')
    
    # Calculate average transaction size per week (using absolute values), now using amount_usd
    df['abs_amount'] = abs(df['amount_usd'])
    weekly_avg = df.groupby('year_week')['abs_amount'].mean().reset_index()
    
    print(f"Analyzed {len(weekly_avg)} weeks of data")
    print(f"Overall average transaction size: ${df['abs_amount'].mean():.2f}")
    
    return weekly_avg

def create_visualizations(monthly_category_spending, top_merchants, weekly_avg, spending_df):
    """Create focused visualizations for the 3 main KPIs in separate windows"""
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. Monthly spending by category (Figure 1)
    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(111)
    
    # Pivot data for stacked bar chart
    pivot_data = monthly_category_spending.pivot(index='year_month', columns='merchant_category', values='spending_amount').fillna(0)
    
    # Create stacked bar chart
    pivot_data.plot(kind='bar', stacked=True, ax=ax1, colormap='Set3')
    ax1.set_title('Total Monthly Spending by Category', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Spending Amount ($)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    
    # Format y-axis to show values in thousands
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    plt.tight_layout()
    fig1.savefig('monthly_spending_by_category.png', dpi=300, bbox_inches='tight')
    
    # 2. Top 3 merchants pie chart (Figure 2)
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    
    top_3_merchants = top_merchants.head(3)
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # Create labels with merchant names and spending amounts
    merchant_labels = [f"{merchant}\n${amount:,.0f}" for merchant, amount in top_3_merchants.items()]
    
    wedges, texts, autotexts = ax2.pie(top_3_merchants.values, 
                                       labels=merchant_labels, 
                                       autopct='%1.1f%%', 
                                       colors=colors, 
                                       startangle=90,
                                       explode=(0.05, 0.05, 0.05))
    
    ax2.set_title('Top 3 Merchants by Total Spend', fontsize=16, fontweight='bold', pad=20)
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
    
    plt.tight_layout()
    fig2.savefig('top_3_merchants.png', dpi=300, bbox_inches='tight')
    
    # 3. Weekly average transaction size (Figure 3)
    fig3 = plt.figure(figsize=(14, 8))
    ax3 = fig3.add_subplot(111)
    
    # Create week labels for better readability (every 10th week)
    week_indices = range(len(weekly_avg))
    
    ax3.plot(week_indices, weekly_avg['abs_amount'], 
             marker='o', linewidth=2, markersize=4, color='darkblue', alpha=0.7, label='Weekly Average')
    ax3.set_title('Average Transaction Size per Week', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Week Number', fontsize=12)
    ax3.set_ylabel('Average Transaction Size ($)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=10)
    
    # Add trend line
    z = np.polyfit(week_indices, weekly_avg['abs_amount'], 1)
    p = np.poly1d(z)
    ax3.plot(week_indices, p(week_indices), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    ax3.legend(fontsize=12)
    
    # Format y-axis to show values in thousands
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.1f}K'))
    
    # Add statistics as text box
    avg_transaction = weekly_avg['abs_amount'].mean()
    min_transaction = weekly_avg['abs_amount'].min()
    max_transaction = weekly_avg['abs_amount'].max()
    
    stats_text = f'''Statistics:
Average: ${avg_transaction:.0f}
Minimum: ${min_transaction:.0f}
Maximum: ${max_transaction:.0f}'''
    
    ax3.text(0.02, 0.98, stats_text, 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    fig3.savefig('weekly_transaction_size.png', dpi=300, bbox_inches='tight')
    
    # Show all figures
    plt.show()
    
    print("Saved individual visualizations:")
    print("- monthly_spending_by_category.png")
    print("- top_3_merchants.png") 
    print("- weekly_transaction_size.png")

def print_summary_report(monthly_category_spending, top_merchants, weekly_avg, spending_df):
    """Print a comprehensive summary report"""
    print("\n" + "="*80)
    print("BANKING TRANSACTIONS ANALYSIS REPORT")
    print("="*80)
    
    # Overall statistics (now using amount_usd)
    total_spending = spending_df['spending_amount'].sum()
    avg_transaction = spending_df['spending_amount'].mean()
    num_transactions = len(spending_df)
    date_range = f"{spending_df['settlement_date'].min().strftime('%Y-%m-%d')} to {spending_df['settlement_date'].max().strftime('%Y-%m-%d')}"
    
    print(f"\nOVERALL STATISTICS (All amounts in USD):")
    print(f"Analysis Period (Settlement Dates): {date_range}")
    print(f"Total Spending: ${total_spending:,.2f}")
    print(f"Number of Transactions: {num_transactions:,}")
    print(f"Average Transaction Size: ${avg_transaction:.2f}")
    
    # Monthly spending by category
    print(f"\n1. MONTHLY SPENDING BY CATEGORY:")
    print("-" * 40)
    top_categories = spending_df.groupby('merchant_category')['spending_amount'].sum().nlargest(5)
    for category, amount in top_categories.items():
        percentage = (amount / total_spending) * 100
        print(f"{category:20s}: ${amount:>10,.2f} ({percentage:5.1f}%)")
    
    # Top merchants
    print(f"\n2. TOP 3 MERCHANTS BY TOTAL SPEND:")
    print("-" * 40)
    for i, (merchant, amount) in enumerate(top_merchants.items(), 1):
        percentage = (amount / total_spending) * 100
        print(f"{i}. {merchant:25s}: ${amount:>10,.2f} ({percentage:5.1f}%)")
    
    # Weekly averages
    print(f"\n3. WEEKLY TRANSACTION SIZE ANALYSIS:")
    print("-" * 40)
    avg_weekly_size = weekly_avg['abs_amount'].mean()
    min_week_avg = weekly_avg['abs_amount'].min()
    max_week_avg = weekly_avg['abs_amount'].max()
    std_weekly = weekly_avg['abs_amount'].std()
    
    print(f"Average Weekly Transaction Size: ${avg_weekly_size:.2f}")
    print(f"Minimum Weekly Average: ${min_week_avg:.2f}")
    print(f"Maximum Weekly Average: ${max_week_avg:.2f}")
    print(f"Standard Deviation: ${std_weekly:.2f}")
    
    # Data quality notes
    print(f"\nDATA QUALITY & ASSUMPTIONS:")
    print("-" * 40)
    print(f"• Only 'clean' transactions included (data_quality_flags = 'clean')")
    print(f"• All amounts converted to USD using fixed exchange rates:")
    print(f"  - EUR to USD: 1.1637")
    print(f"  - GBP to USD: 1.3623") 
    print(f"  - CHF to USD: 1.2421")
    print(f"  - JPY to USD: 0.00688 (1/145.41)")
    print(f"• Analysis based on settlement_date (not transaction_date)")
    print(f"• Spending defined as negative amounts (outgoing transactions)")
    print(f"• Missing merchant names filled as 'Unknown Merchant'")
    print(f"• Missing categories filled as 'Uncategorized'")
    
    print("\n" + "="*80)

def main():
    """Main analysis function"""
    try:
        # Load and clean data
        df = load_and_clean_data('cleaned_banking_transactions.csv')
        
        # Calculate KPIs
        monthly_category_spending, spending_df = calculate_monthly_spending_by_category(df)
        top_merchants, all_merchants = get_top_merchants(df)
        weekly_avg = calculate_weekly_avg_transaction_size(df)
        
        # Create visualizations
        create_visualizations(monthly_category_spending, all_merchants, weekly_avg, spending_df)
        
        # Print summary report
        print_summary_report(monthly_category_spending, top_merchants, weekly_avg, spending_df)
        
        print(f"\nAnalysis complete! Individual KPI visualizations have been saved and displayed.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please ensure 'cleaned_banking_transactions.csv' exists in the current directory.")

if __name__ == "__main__":
    main() 