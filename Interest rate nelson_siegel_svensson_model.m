function nelson_siegel_svensson_model()
    % Sample data: maturities (in years) and corresponding yields (in %)
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30];
    yields = [0.25, 0.5, 1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]; % Example yields

    % Initial parameter guess for [beta0, beta1, beta2, beta3, lambda1, lambda2]
    initialParams = [3, -2, 2, -1, 1, 0.5];

    % Optimization options
    options = optimoptions('lsqnonlin', 'Display', 'iter', 'TolFun', 1e-6, 'TolX', 1e-6);

    % Fit the model using nonlinear least squares optimization
    params = lsqnonlin(@(p) nss_residuals(p, maturities, yields), initialParams, [], [], options);

    % Display the fitted parameters
    disp('Fitted parameters:');
    disp(params);

    % Plot the observed yields and the fitted curve
    fittedYields = nss_yield_curve(params, maturities);
    figure;
    plot(maturities, yields, 'o', 'DisplayName', 'Observed Yields');
    hold on;
    plot(maturities, fittedYields, 'r-', 'DisplayName', 'Fitted NSS Curve');
    xlabel('Maturity (years)');
    ylabel('Yield (%)');
    legend;
    title('Nelson-Siegel-Svensson Yield Curve Fitting');
    grid on;
end

function residuals = nss_residuals(params, maturities, yields)
    % Calculate the residuals between observed yields and model-predicted yields
    predictedYields = nss_yield_curve(params, maturities);
    residuals = yields - predictedYields;
end

function yields = nss_yield_curve(params, maturities)
    % Nelson-Siegel-Svensson model
    beta0 = params(1);
    beta1 = params(2);
    beta2 = params(3);
    beta3 = params(4);
    lambda1 = params(5);
    lambda2 = params(6);

    term1 = (1 - exp(-lambda1 * maturities)) ./ (lambda1 * maturities);
    term2 = term1 - exp(-lambda1 * maturities);
    term3 = (1 - exp(-lambda2 * maturities)) ./ (lambda2 * maturities) - exp(-lambda2 * maturities);

    yields = beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3;
end
