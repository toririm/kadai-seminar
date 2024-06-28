fn main() {
    println!(
        "cargo:rustc-link-search=native={}",
        std::env::var("VCPKG_ROOT").unwrap() + r"\installed\x64-windows\lib"
    );
    println!("cargo:rustc-link-lib=dylib=openblas");
}
