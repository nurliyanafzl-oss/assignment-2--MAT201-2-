import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

# 1. Page Configuration
st.set_page_config(page_title="Calculus MAT201 - Gradient Visualizer", layout="wide")

st.title("Gradient & Direction of Steepest Ascent")
st.write("Visualizing how partial derivatives define the path of maximum increase.")

# 2. Sidebar for User Input
st.sidebar.header("Input Settings")
equation_input = st.sidebar.text_input("Enter function f(x, y):", "x**2 - y**2")
x_val = st.sidebar.slider("Point x:", -5.0, 5.0, 1.0)
y_val = st.sidebar.slider("Point y:", -5.0, 5.0, 1.0)

try:
    # 3. Symbolic Math Calculations
    x, y = sp.symbols('x y')
    f_sym = sp.sympify(equation_input)
    
    # Calculate Partial Derivatives
    fx = sp.diff(f_sym, x)
    fy = sp.diff(f_sym, y)
    
    # Evaluate at the specific point
    f_at_pt = float(f_sym.subs({x: x_val, y: y_val}))
    dfx_val = float(fx.subs({x: x_val, y: y_val}))
    dfy_val = float(fy.subs({x: x_val, y: y_val}))

    # 4. Display Step-by-Step Calculations
    st.subheader("Step-by-Step Gradient Calculation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. Partial Derivatives:**")
        st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(fx)}")
        st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(fy)}")
        
    with col2:
        st.markdown("**2. Gradient Vector at Point:**")
        st.latex(rf"\nabla f({x_val}, {y_val}) = \langle {dfx_val:.2f}, {dfy_val:.2f} \rangle")
        st.write(f"This vector points in the direction of steepest ascent at $z = {f_at_pt:.2f}$")

    # 5. 3D Visualization Logic
    # Generate grid
    x_range = np.linspace(-5, 5, 50)
    y_range = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x_range, y_range)
    f_func = sp.lambdify((x, y), f_sym, "numpy")
    Z = f_func(X, Y)

    # Normalize the gradient vector for better visualization
    mag = np.sqrt(dfx_val**2 + dfy_val**2)
    if mag > 0:
        # Scale the arrow so it's always visible (1.5 units long)
        ux, uy = (dfx_val/mag)*1.5, (dfy_val/mag)*1.5
    else:
        ux, uy = 0, 0

    # 6. Create Plotly Figure
    fig = go.Figure()

    # Surface Plot
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8, name="Surface"))

    # Gradient Arrow Shaft (Yellow Line)
    fig.add_trace(go.Scatter3d(
        x=[x_val, x_val + ux], y=[y_val, y_val + uy], z=[f_at_pt, f_at_pt],
        mode='lines', line=dict(color='yellow', width=10), name="Direction of Steepest Ascent"
    ))

    # Gradient Arrow Head (Cone)
    fig.add_trace(go.Cone(
        x=[x_val + ux], y=[y_val + uy], z=[f_at_pt],
        u=[ux], v=[uy], w=[0],
        sizemode="absolute", sizeref=0.5, showscale=False, colorscale=[[0, 'yellow'], [1, 'yellow']]
    ))

    fig.update_layout(scene=dict(aspectmode='cube'), height=700)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}. Please use Python syntax (e.g., x**2 for $x^2$).")
