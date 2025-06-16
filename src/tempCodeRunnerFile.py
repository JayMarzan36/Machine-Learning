    show_3d_plot(
        weights=final_weight,
        bias=final_bias,
        modal_function=compute_prediction,
        feature_x_index=0,  # First variable
        feature_y_index=1,  # Second variable
        input_range=[0, 10],  # Range of input values
        resolution=100,  # Higher = smoother surface
        fixed_values=[0.0, 0.0, 0.0],  # Total features = 3 in this case
    )