use ndarray::{Array, Array1};
use num::complex::Complex;
use plotters::prelude::*;

// Initialize the wavefunction
fn initialize_wf(xj: &Array1<f64>, x0: f64, k0: f64, sigma0: f64) -> Array1<Complex<f64>> {
    xj.mapv(|x| {
        Complex::new(0.0, k0 * (x - x0)).exp()
            * Complex::new(-0.5 * (x - x0).powi(2) / sigma0.powi(2), 0.0).exp()
    })
}

// Initialize potential
fn initialize_vpot(xj: &Array1<f64>) -> Array1<f64> {
    let k0 = 1.0;
    xj.mapv(|x| 0.5 * k0 * x.powi(2))
}

// Operate the Hamiltonian on the wavefunction
fn ham_wf(wf: &Array1<Complex<f64>>, vpot: &Array1<f64>, dx: f64) -> Array1<Complex<f64>> {
    let n = wf.len();
    let mut hwf = Array::zeros(n);

    for i in 1..n - 1 {
        hwf[i] = -0.5 * (wf[i + 1] - 2.0 * wf[i] + wf[i - 1]) / dx.powi(2);
    }

    hwf[0] = -0.5 * (wf[1] - 2.0 * wf[0]) / dx.powi(2);
    hwf[n - 1] = -0.5 * (-2.0 * wf[n - 1] + wf[n - 2]) / dx.powi(2);

    hwf + vpot.mapv(Complex::from) * wf
}

// Time propagation from t to t + dt
fn time_propagation(
    wf: &Array1<Complex<f64>>,
    vpot: &Array1<f64>,
    dx: f64,
    dt: f64,
) -> Array1<Complex<f64>> {
    let mut wf = wf.clone();
    let mut twf = wf.clone();
    let mut hwf;
    let mut zfact = Complex::new(1.0, 0.0);

    for iexp in 1..5 {
        zfact *= Complex::new(0.0, -dt) / Complex::new(iexp as f64, 0.0);
        hwf = ham_wf(&twf, vpot, dx);
        wf = wf + zfact * &hwf;
        twf = hwf;
    }

    wf
}

fn main() {
    // Initial wavefunction parameters
    let x0 = -2.0;
    let k0 = 0.0;
    let sigma0 = 1.0;

    // Time propagation parameters
    let tprop = 40.0;
    let dt = 0.005;
    let nt = (tprop / dt) as usize + 1;

    // Set the coordinate
    let xmin = -10.0;
    let xmax = 10.0;
    let n = 250;

    let dx = (xmax - xmin) / (n as f64 + 1.0);
    let xj = Array::linspace(xmin + dx, xmax - dx, n);

    // Initialize the wavefunction and potential
    let mut wf = initialize_wf(&xj, x0, k0, sigma0);
    let vpot = initialize_vpot(&xj);

    let mut wavefunctions = Vec::with_capacity(nt);
    for it in 0..nt {
        if it % (nt / 100) == 0 {
            wavefunctions.push(wf.clone());
        }
        wf = time_propagation(&wf, &vpot, dx, dt);
        println!("{} {}", it, nt);
    }

    // Create the animation
    let root = BitMapBackend::gif("wavefunctions.gif", (800, 600), 150)
        .unwrap()
        .into_drawing_area();

    for (i, wf) in wavefunctions.iter().enumerate() {
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Wavefunction Animation Frame {}", i),
                ("sans-serif", 20).into_font(),
            )
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-5.0..5.0, -1.2..5.0)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                xj.iter().zip(wf.iter().map(|x| x.re)).map(|(&x, y)| (x, y)),
                &RED,
            ))
            .unwrap()
            .label("Real part of ψ(x)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .draw_series(LineSeries::new(
                xj.iter().zip(wf.iter().map(|x| x.im)).map(|(&x, y)| (x, y)),
                &BLUE,
            ))
            .unwrap()
            .label("Imaginary part of ψ(x)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .draw_series(LineSeries::new(
                xj.iter()
                    .zip(wf.iter().map(|x| x.norm()))
                    .map(|(&x, y)| (x, y)),
                &GREEN,
            ))
            .unwrap()
            .label("|ψ(x)|")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        chart
            .draw_series(LineSeries::new(
                xj.iter().zip(vpot.iter()).map(|(&x, &y)| (x, y)),
                &BLACK,
            ))
            .unwrap()
            .label("V(x)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw()
            .unwrap();

        root.present().unwrap();
    }

    println!("Animation saved as wavefunctions.gif");
}
