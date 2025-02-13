import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import base64
from io import BytesIO
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Trading Journal Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'edit_index' not in st.session_state:
    st.session_state.edit_index = None

# Functions for image handling
def save_image(image_file):
    if image_file is not None:
        bytes_data = image_file.getvalue()
        base64_str = base64.b64encode(bytes_data).decode()
        return base64_str
    return None

def display_image(base64_str):
    if base64_str:
        st.image(BytesIO(base64.b64decode(base64_str)))

# Function to calculate trade statistics
def calculate_stats(trades_df):
    if trades_df.empty:
        return {}
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] <= 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        'total_pnl': trades_df['pnl'].sum(),
        'largest_win': trades_df['pnl'].max(),
        'largest_loss': trades_df['pnl'].min()
    }

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Daily Journal", "Weekly Analysis", "Trade History", "Performance Metrics"])

if page == "Daily Journal":
    st.title("📈 Daily Trading Journal")
    
    # Edit mode handling
    if st.session_state.edit_mode:
        trade_to_edit = st.session_state.trades[st.session_state.edit_index]
        st.info("Editing trade for " + trade_to_edit['symbol'])
    
    # Input form
    with st.form("trade_form"):
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            date = st.date_input("Date", 
                               value=datetime.strptime(trade_to_edit['date'], "%Y-%m-%d").date() if st.session_state.edit_mode else datetime.today())
            symbol = st.text_input("Symbol", value=trade_to_edit['symbol'] if st.session_state.edit_mode else "")
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01, 
                                        value=float(trade_to_edit['entry_price']) if st.session_state.edit_mode else 0.0)
            
        with col2:
            exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01,
                                       value=float(trade_to_edit['exit_price']) if st.session_state.edit_mode else 0.0)
            position_size = st.number_input("Position Size", min_value=1,
                                          value=int(trade_to_edit['position_size']) if st.session_state.edit_mode else 1)
            trade_type = st.selectbox("Trade Type", ["Long", "Short"],
                                    index=0 if not st.session_state.edit_mode or trade_to_edit['trade_type'] == "Long" else 1)
            
        with col3:
            risk_reward = st.number_input("Risk/Reward Ratio", min_value=0.0, step=0.1,
                                        value=float(trade_to_edit['risk_reward']) if st.session_state.edit_mode else 0.0)
            trade_tags = st.multiselect("Trade Tags", 
                                      ["Breakout", "Reversal", "Trend Following", "Counter-Trend", "News Based", "Technical", "Fundamental"],
                                      default=trade_to_edit.get('tags', []) if st.session_state.edit_mode else [])
            
        setup_quality = st.slider("Setup Quality (1-10)", 1, 10, 
                                value=trade_to_edit.get('setup_quality', 5) if st.session_state.edit_mode else 5)
        
        col1, col2 = st.columns(2)
        with col1:
            pre_market_plan = st.text_area("Pre-market Plan",
                                         value=trade_to_edit['pre_market_plan'] if st.session_state.edit_mode else "")
        with col2:
            trade_notes = st.text_area("Trade Notes",
                                     value=trade_to_edit['trade_notes'] if st.session_state.edit_mode else "")
        
        emotions = st.select_slider("Emotional State During Trade",
                                  options=["Very Anxious", "Anxious", "Neutral", "Confident", "Very Confident"],
                                  value=trade_to_edit.get('emotions', 'Neutral') if st.session_state.edit_mode else 'Neutral')
        
        chart_url = st.text_input("Chart URL/Link",
                                 value=trade_to_edit['chart_url'] if st.session_state.edit_mode else "")
        
        uploaded_file = st.file_uploader("Upload Trade Screenshot", type=['png', 'jpg', 'jpeg'])
        
        if st.session_state.edit_mode:
            submit_button = st.form_submit_button("Update Trade")
        else:
            submit_button = st.form_submit_button("Add Trade")
        
        if submit_button:
            if not symbol:
                st.error("Please enter a symbol")
            else:
                trade = {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "position_size": position_size,
                    "trade_type": trade_type,
                    "risk_reward": risk_reward,
                    "pre_market_plan": pre_market_plan,
                    "trade_notes": trade_notes,
                    "chart_url": chart_url,
                    "pnl": (exit_price - entry_price) * position_size if trade_type == "Long" 
                           else (entry_price - exit_price) * position_size,
                    "tags": trade_tags,
                    "setup_quality": setup_quality,
                    "emotions": emotions
                }
                
                if uploaded_file:
                    trade['screenshot'] = save_image(uploaded_file)
                elif st.session_state.edit_mode and 'screenshot' in trade_to_edit:
                    trade['screenshot'] = trade_to_edit['screenshot']
                
                if st.session_state.edit_mode:
                    st.session_state.trades[st.session_state.edit_index] = trade
                    st.session_state.edit_mode = False
                    st.session_state.edit_index = None
                    st.success("Trade updated successfully!")
                else:
                    st.session_state.trades.append(trade)
                    st.success("Trade added successfully!")

elif page == "Weekly Analysis":
    st.title("📊 Weekly Analysis")
    
    if st.session_state.trades:
        df = pd.DataFrame(st.session_state.trades)
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.isocalendar().year
        
        # Weekly PnL
        weekly_pnl = df.groupby(['year', 'week'])['pnl'].sum().reset_index()
        weekly_pnl['week_label'] = weekly_pnl['year'].astype(str) + '-W' + weekly_pnl['week'].astype(str)
        
        fig_weekly = px.bar(weekly_pnl, x='week_label', y='pnl',
                           title='Weekly P&L',
                           labels={'week_label': 'Week', 'pnl': 'Profit/Loss'})
        fig_weekly.update_layout(showlegend=False)
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Weekly Statistics
        col1, col2, col3, col4 = st.columns(4)
        current_week = datetime.today().isocalendar().week
        current_year = datetime.today().isocalendar().year
        weekly_trades = df[
            (df['date'].dt.isocalendar().week == current_week) & 
            (df['date'].dt.isocalendar().year == current_year)
        ]
        
        weekly_stats = calculate_stats(weekly_trades)
        
        col1.metric("Week's Trades", weekly_stats.get('total_trades', 0))
        col2.metric("Week's Win Rate", f"{weekly_stats.get('win_rate', 0):.1f}%")
        col3.metric("Week's P&L", f"₹{weekly_stats.get('total_pnl', 0):,.2f}")
        col4.metric("Week's Largest Trade", f"₹{weekly_stats.get('largest_win', 0):,.2f}")
        
        # Tag Analysis
        st.subheader("Performance by Tag")
        if 'tags' in df.columns:
            all_tags = []
            for tags in df['tags']:
                all_tags.extend(tags)
            unique_tags = list(set(all_tags))
            
            tag_performance = []
            for tag in unique_tags:
                tag_trades = df[df['tags'].apply(lambda x: tag in x)]
                tag_stats = calculate_stats(tag_trades)
                tag_performance.append({
                    'tag': tag,
                    'win_rate': tag_stats['win_rate'],
                    'total_pnl': tag_stats['total_pnl'],
                    'trade_count': tag_stats['total_trades']
                })
            
            tag_df = pd.DataFrame(tag_performance)
            fig_tags = go.Figure()
            fig_tags.add_trace(go.Bar(name='Win Rate', x=tag_df['tag'], y=tag_df['win_rate']))
            fig_tags.add_trace(go.Bar(name='Trade Count', x=tag_df['tag'], y=tag_df['trade_count']))
            fig_tags.update_layout(barmode='group', title='Tag Performance Analysis')
            st.plotly_chart(fig_tags, use_container_width=True)

elif page == "Trade History":
    st.title("📝 Trade History")
    
    if st.session_state.trades:
        df = pd.DataFrame(st.session_state.trades)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Start Date", df['date'].min())
        with col2:
            end_date = st.date_input("End Date", df['date'].max())
        with col3:
            symbol_filter = st.multiselect("Filter by Symbol", options=sorted(df['symbol'].unique()))
        
        # Apply filters
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        if symbol_filter:
            mask = mask & (df['symbol'].isin(symbol_filter))
        filtered_df = df.loc[mask]
        
        # Display trades
        for index, trade in filtered_df.iterrows():
            with st.expander(f"{trade['date'].strftime('%Y-%m-%d')} - {trade['symbol']} ({trade['trade_type']})"):
                col1, col2, col3 = st.columns([2,2,1])
                
                with col1:
                    st.write(f"Entry Price: ₹{trade['entry_price']:.2f}")
                    st.write(f"Exit Price: ₹{trade['exit_price']:.2f}")
                    st.write(f"Position Size: {trade['position_size']}")
                    st.write(f"Tags: {', '.join(trade['tags'])}")
                    
                with col2:
                    st.write(f"P&L: ₹{trade['pnl']:.2f}")
                    st.write(f"Risk/Reward: {trade['risk_reward']:.2f}")
                    st.write(f"Setup Quality: {trade['setup_quality']}/10")
                    st.write(f"Emotional State: {trade['emotions']}")
                
                with col3:
                    edit_col, delete_col = st.columns(2)
                    with edit_col:
                        if st.button("✏️ Edit", key=f"edit_{index}"):
                            st.session_state.edit_mode = True
                            st.session_state.edit_index = index
                            st.rerun()  # Changed from experimental_rerun()
                    with delete_col:
                        if st.button("🗑️ Delete", key=f"delete_{index}"):
                            st.session_state.trades.pop(index)
                            st.rerun()  # Changed from experimental_rerun()

                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Pre-market Plan:")
                    st.write(trade['pre_market_plan'])
                
                with col2:
                    st.write("Trade Notes:")
                    st.write(trade['trade_notes'])
                
                if trade.get('screenshot'):
                    st.write("Trade Screenshot:")
                    display_image(trade['screenshot'])
                
                if trade['chart_url']:
                    st.write("Chart Link:")
                    st.write(trade['chart_url'])

# ... (previous code remains the same until the Performance Metrics page)

elif page == "Performance Metrics":
    st.title("📈 Performance Metrics")
    
    if st.session_state.trades:
        df = pd.DataFrame(st.session_state.trades)
        df['date'] = pd.to_datetime(df['date'])
        
        # Time period selector
        time_period = st.selectbox("Select Time Period", 
                                 ["All Time", "This Year", "This Month", "Last 30 Days", "Last 90 Days"])
        
        # Filter data based on selected time period
        if time_period == "This Year":
            df = df[df['date'].dt.year == datetime.today().year]
        elif time_period == "This Month":
            df = df[
                (df['date'].dt.year == datetime.today().year) & 
                (df['date'].dt.month == datetime.today().month)
            ]
        elif time_period == "Last 30 Days":
            df = df[df['date'] >= (datetime.today() - pd.Timedelta(days=30))]
        elif time_period == "Last 90 Days":
            df = df[df['date'] >= (datetime.today() - pd.Timedelta(days=90))]
        
        # Calculate metrics
        stats = calculate_stats(df)
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total P&L", f"₹{stats['total_pnl']:,.2f}")
        col2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        col3.metric("Total Trades", stats['total_trades'])
        col4.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Average Win", f"₹{stats['avg_win']:,.2f}")
        col2.metric("Average Loss", f"₹{stats['avg_loss']:,.2f}")
        col3.metric("Largest Win", f"₹{stats['largest_win']:,.2f}")
        col4.metric("Largest Loss", f"₹{stats['largest_loss']:,.2f}")
        
        # Cumulative P&L Chart
        st.subheader("Cumulative P&L")
        df['cumulative_pnl'] = df['pnl'].cumsum()
        fig_cum_pnl = px.line(df, x='date', y='cumulative_pnl',
                             title='Cumulative Profit/Loss Over Time')
        fig_cum_pnl.update_traces(line_color='#4CAF50')
        st.plotly_chart(fig_cum_pnl, use_container_width=True)
        
        # Win/Loss Distribution
        st.subheader("Win/Loss Distribution")
        fig_dist = px.histogram(df, x='pnl', nbins=20,
                               title='Distribution of Trade P&L',
                               color_discrete_sequence=['#4CAF50'])
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Performance by Day of Week
        st.subheader("Performance by Day of Week")
        df['day_of_week'] = df['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        daily_perf = df.groupby('day_of_week')['pnl'].agg(['sum', 'count', 'mean']).reindex(day_order)
        
        fig_dow = go.Figure(data=[
            go.Bar(name='Total P&L', y=daily_perf['sum']),
            go.Bar(name='Average P&L', y=daily_perf['mean'])
        ])
        fig_dow.update_layout(xaxis={'categoryorder':'array', 
                                   'categoryarray':day_order},
                            title='P&L by Day of Week',
                            barmode='group')
        st.plotly_chart(fig_dow, use_container_width=True)
        
        # Performance by Setup Quality
        if 'setup_quality' in df.columns:
            st.subheader("Performance by Setup Quality")
            setup_perf = df.groupby('setup_quality').agg({
                'pnl': ['mean', 'count', 'sum'],
                'trade_type': 'count'
            }).reset_index()
            
            fig_setup = go.Figure(data=[
                go.Bar(name='Average P&L', x=setup_perf['setup_quality'], 
                      y=setup_perf['pnl']['mean']),
                go.Bar(name='Trade Count', x=setup_perf['setup_quality'], 
                      y=setup_perf['pnl']['count'])
            ])
            fig_setup.update_layout(title='Performance by Setup Quality Rating',
                                  barmode='group')
            st.plotly_chart(fig_setup, use_container_width=True)
        
        # Export functionality
        st.subheader("Export Data")
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="trading_journal_export.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
    else:
        st.info("No trades recorded yet. Add some trades in the Daily Journal!")
