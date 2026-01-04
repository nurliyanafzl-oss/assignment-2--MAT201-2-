import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

# 1. Page Configuration
st.set_page_config(page_title="MAT201 - Gradient Visualizer", layout="wide")

# 2. Application Title & Description
st.title("Gradient and Direction of Steepest Ascent Visualizer")
st.write("This application visualizes the gradient vector $\\nabla f(x, y)$ on a 3D surface. The arrow always points towards the steepest ascent.")

# 3. Sidebar for User Input
st.sidebar.header("User Input Settings")
# Default function set to sin(x)*cos(y) to test the scaling logic
equation_input = st.sidebar.text_input("Enter function f(x, y):", "sin(x)*cos(y)")
x_coord = st.sidebar.slider("Point x:", -5.0, 5.0, 1.0)
y_coord = st.sidebar.slider("Point y:", -5.0, 5.0, 1.0)

try:
    # 4. Mathematical Calculations using Sympy
    x, y = sp.symbols('x y')
    f_sym = sp.sympify(equation_input)
    
    # Partial derivatives
    fx = sp.diff(f_sym, x)
    fy = sp.diff(f_sym, y)
    
    # Evaluate values at the specific point
    grad_x_val = float(fx.subs({x: x_coord, y: y_coord}))
    grad_y_val = float(fy.subs({x: x_coord, y: y_coord}))
    z_coord = float(f_sym.subs({x: x_coord, y: y_coord}))

    # --- AUTO-SCALE LOGIC (Normalization) ---
    # This ensures the arrow is visible even for very small gradients
    magnitude = np.sqrt(grad_x_val**2 + grad_y_val**2)
    
    if magnitude > 0:
        # Normalize and set a fixed display length of 1.5 units for the visual
        display_x = (grad_x_val / magnitude) * 1.5
        display_y = (grad_y_val / magnitude) * 1.5
    else:
        display_x, display_y = 0, 0
    # ---------------------------------------

    # 5. Display Mathematical Results
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Mathematical Expression")
        st.latex(rf"f(x, y) = {sp.latex(f_sym)}")
    with col2:
        st.write("### Gradient at Point")
        st.latex(rf"\nabla f({x_coord}, {y_coord}) = \langle {grad_x_val:.2f}, {grad_y_val:.2f} \rangle")

    # 6. Create 3D Surface Data
    x_range = np.linspace(-5, 5, 50)
    y_range = np.linspace(-5, 5, 50)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    f_numpy = sp.lambdify((x, y), f_sym, 'numpy')
    Z_grid = f_numpy(X_grid, Y_grid)

    # 7. Build Plotly Figure
    fig = go.Figure()

    # Add Surface Plot
    fig.add_trace(go.Surface(
        z=Z_grid, x=X_grid, y=Y_grid, 
        colorscale='Viridis', 
        opacity=0.7,
        showscale=True
    ))

    # Add the Gradient Arrow Shaft (The Line)
    fig.add_trace(go.Scatter3d(
        x=[x_coord, x_coord + display_x],
        y=[y_coord, y_coord + display_y],
        z=[z_coord, z_coord],
        mode='lines',
        line=dict(color='yellow', width=12),
        name="Gradient Shaft"
    ))

    # Add the Gradient Arrow Head (The Cone)
    fig.add_trace(go.Cone(
        x=[x_coord + display_x],
        y=[y_coord + display_y],
        z=[z_coord],
        u=[display_x],
        v=[display_y],
        w=[0],
        sizemode="absolute",
        sizeref=0.6,
        colorscale=[[0, 'yellow'], [1, 'yellow']],
        showscale=False,
        name="Arrow Head"
    ))

    # 8. Update Layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='f(x,y)',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}. Please ensure you use Python syntax (e.g., x**2 instead of x^2).")

st.markdown("---")
st.caption("Calculus MAT201 Assignment | Developed with Streamlit & Plotly")
