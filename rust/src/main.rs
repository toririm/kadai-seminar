use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use num::complex::Complex;
use plotters::prelude::*;

fn init_wf(
    xj: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    x0: f64,
    k0: f64,
    sigma0: f64,
) -> ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 1]>> {
    let mut wf = Array::zeros(xj.len());
    for i in 0..xj.len() {
        wf[i] = Complex::new(0.0, k0 * (xj[i] - x0)).exp()
            * Complex::new(-0.5 * (xj[i] - x0).powi(2) / sigma0.powi(2), 0.0);
    }
    wf
}

fn init_vpot(
    xj: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
) -> ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 1]>> {
    let k0 = 1.0;
    let mut vpot = Array::zeros(xj.len());
    for i in 0..xj.len() {
        vpot[i] = Complex::new(0.0, 0.5 * k0 * xj[i].powi(2));
    }
    vpot
}

fn ham_wf(
    wf: &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 1]>>,
    vpot: &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 1]>>,
    dx: f64,
) -> ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 1]>> {
    let mut hwf = Array::zeros(wf.len());
    for i in 1..(wf.len() - 1) {
        hwf[i] = -0.5 * (wf[i + 1] - 2.0 * wf[i] + wf[i - 1]) / dx.powi(2)
    }

    let mut i = 0;
    hwf[i] = -0.5 * (wf[i + 1] - 2.0 * wf[i]) / dx.powi(2);

    i = wf.len() - 1;
    hwf[i] = -0.5 * (-2.0 * wf[i] + wf[i - 1]) / dx.powi(2);

    hwf = hwf + vpot.dot(wf);

    hwf
}

fn time_propagation(
    wf: &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 1]>>,
    vpot: &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 1]>>,
    dx: f64,
    dt: f64,
) -> ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 1]>> {
    let mut twf;
    let mut hwf;
    let mut cur_wf = wf.clone();

    twf = wf.clone();
    let mut zfact = Complex::new(1.0, 0.0);

    for iexp in 1..5 {
        zfact *= Complex::new(0.0, -dt) / Complex::new(iexp as f64, 0.0);
        hwf = ham_wf(&twf, &vpot, dx);
        cur_wf = cur_wf + zfact * hwf.clone();
        twf = hwf.clone();
    }

    cur_wf
}

fn main() {
    println!("Hello, world!");
    // initial wavefunction parameters
    let x0 = -2.0;
    let k0 = 0.0;
    let sigma0 = 1.0;

    // time propagation parameters
    let tprop = 40.0;
    let dt = 0.005;
    let nt = ((tprop / dt) + 1.0) as i32;

    // set the coordinate
    let xmin = -10.0;
    let xmax = 10.0;
    let n = 250;

    let dx = (xmax - xmin) / (n as f64);
    let mut xj: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = Array::zeros(n);

    for i in 0..n {
        xj[i] = xmin + (i as f64) * dx;
    }

    let mut wf = init_wf(&xj, x0, k0, sigma0);
    let vpot = init_vpot(&xj);

    let mut wavefunctions = Vec::new();
    for it in 0..=nt {
        if it % (nt / 100) as i32 == 0 {
            wavefunctions.push(wf.clone());
        }

        wf = time_propagation(&wf, &vpot, dx, dt);
        println!("{} {}", it, nt);
    }

    let area = BitMapBackend::gif("wavefunctions.gif", (800, 600), 150)
        .unwrap()
        .into_drawing_area();

    for (i, wf) in wavefunctions.iter().enumerate() {
        area.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&area)
            .caption(
                format!("Wavefunction at t = {:.2}", i as f64 * dt),
                ("sans-serif", 20).into_font(),
            )
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(xj[0]..xj[n - 1], -1.0..1.0)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                xj.iter().zip(wf.iter()).map(|(x, y)| (*x, y.re)),
                &RED,
            ))
            .unwrap();
    }
}
